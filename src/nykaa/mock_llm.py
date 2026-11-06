"""Deterministic offline LLM backend for the CrewAI crew (MOCK_LLM mode).

Subclasses ``crewai.llms.base_llm.BaseLLM`` -- CrewAI's own documented extension
point for a non-litellm LLM. Zero API keys, zero network.

Verified behaviour of crewai 1.15.36 (see the probe notes in README.md)
-----------------------------------------------------------------------
* ``BaseLLM`` is a **Pydantic model**, so extra state is declared as model
  fields, not ``__init__`` parameters.
* Exactly one abstract method must be implemented: ``call``.
* ``messages`` arrives as a **list of plain dicts**, not ``LLMMessage`` objects.
* CrewAI decides native-vs-text tool calling in ``check_native_tool_support``
  via ``hasattr(llm, "supports_function_calling")``. We deliberately return
  ``False`` to take the **text/ReAct path**, which is the path the brief's two
  documented pitfalls are about -- so this implementation exercises and handles
  both of them explicitly.

The two documented pitfalls, and how each is handled here
---------------------------------------------------------
1. **The ``Observation:`` trap.** CrewAI's own system prompt contains the
   example text ``Observation: the result of the action``. Any parser that
   searches the *whole conversation* for ``"Observation:"`` matches that
   template on the very first call, before any tool has run, and returns
   placeholder text as the final answer.
   -> :func:`_observation_from_own_output` reads **only the assistant-authored
   messages** (our own generated text), never the system prompt.

2. **The tool-name trap.** Dispatching a tool call by testing whether a name
   such as ``"lookup"`` appears in the tool name silently misroutes a tool
   literally called ``rag_lookup``.
   -> :func:`_arguments_from_schema` dispatches off the tool's **declared
   argument schema** (its required keys), and never inspects the tool's name.
"""

from __future__ import annotations

import json
import re
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import Field

ORD_ID_RE = re.compile(r"\bORD-\d{3,6}\b", re.IGNORECASE)


class ToolArgumentsUnavailable(Exception):
    """Raised when a task does not supply what the tool's schema requires.

    This is a normal, expected outcome -- a policy-only question carries no
    order id, so the lookup tool must simply not be called rather than error.
    """


# --------------------------------------------------------------------------- #
# Message helpers
# --------------------------------------------------------------------------- #


def _as_dicts(messages: Any) -> list[dict[str, str]]:
    """Normalise whatever CrewAI hands us into ``[{"role":..., "content":...}]``."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out: list[dict[str, str]] = []
    for m in messages or []:
        if isinstance(m, dict):
            out.append({"role": str(m.get("role", "")), "content": str(m.get("content", "") or "")})
        else:
            out.append(
                {
                    "role": str(getattr(m, "role", "")),
                    "content": str(getattr(m, "content", "") or ""),
                }
            )
    return out


def _current_task_text(msgs: list[dict[str, str]]) -> str:
    """Extract the task description from the user message."""
    for m in msgs:
        if m["role"] == "user" and "Current Task:" in m["content"]:
            after = m["content"].split("Current Task:", 1)[1]
            return after.split("\n", 1)[0].strip()
    for m in reversed(msgs):
        if m["role"] == "user":
            return m["content"].strip()
    return ""


def _full_user_text(msgs: list[dict[str, str]]) -> str:
    """All user-authored content, including prior tasks' outputs.

    The composer needs the *whole* prompt: CrewAI injects the context tasks'
    outputs into the same user message, below the ``Current Task:`` line. Only
    the composer path uses this; tool dispatch still uses just the task line.
    """
    return "\n".join(m["content"] for m in msgs if m["role"] == "user")


def _observation_from_own_output(msgs: list[dict[str, str]]) -> str | None:
    """Return the tool result, read ONLY from our own generated text.

    Pitfall 1 lives here. We restrict the search to ``role == "assistant"``
    messages -- text this LLM produced -- so the ``Observation:`` example inside
    CrewAI's *system* prompt template can never be mistaken for a tool result.
    """
    for m in reversed(msgs):
        if m["role"] != "assistant":
            continue
        content = m["content"]
        if "Observation:" in content:
            return content.split("Observation:", 1)[1].strip()
        return None
    return None


def _question_from_task(task_text: str) -> str:
    """Strip the instruction prefix so the retrieval query is the real question.

    Task descriptions carry framing for the agent ("Answer this ... using the
    rag_lookup tool. Question: <q>"). Embedding the framing dilutes the query
    vector and drags in unrelated chunks, so the tool is given only the question.
    """
    if "Question:" in task_text:
        return task_text.split("Question:", 1)[1].strip()
    return task_text.strip()


def _schema_required_keys(tool: Any) -> list[str]:
    """Required argument names from a tool's declared schema."""
    schema = getattr(tool, "args_schema", None)
    if schema is not None and hasattr(schema, "model_json_schema"):
        js = schema.model_json_schema()
        if isinstance(js, dict) and "required" in js:
            return list(js["required"])
    args = getattr(tool, "args", None) or {}
    return [k for k, v in args.items() if getattr(v, "required", False)]


def _arguments_from_schema(tool: Any, task_text: str) -> dict[str, Any]:
    """Build tool arguments from the tool's DECLARED SCHEMA.

    Pitfall 2 lives here. The tool's *name* is never inspected -- routing is
    driven entirely by which argument names the schema declares as required.
    A tool named ``rag_lookup`` is therefore indistinguishable from one named
    ``policy_search``, which is the point.
    """
    required = _schema_required_keys(tool)

    if "query" in required:
        return {"query": _question_from_task(task_text)}

    if "record_id" in required:
        match = ORD_ID_RE.search(task_text)
        if not match:
            raise ToolArgumentsUnavailable(
                "tool schema requires 'record_id' but no ORD-#### id appears in "
                f"the task text: {task_text!r}"
            )
        return {"record_id": match.group(0).upper()}

    raise ValueError(
        f"no dispatch rule for declared schema keys {required!r}; add one rather "
        "than matching on the tool's name"
    )


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Find every top-level JSON object embedded in ``text``."""
    found: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    i = 0
    while True:
        start = text.find("{", i)
        if start == -1:
            return found
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        i = end


# --------------------------------------------------------------------------- #
# The mock LLM
# --------------------------------------------------------------------------- #


class MockLLM(BaseLLM):
    """Deterministic, offline, ReAct-path LLM."""

    calls: int = 0
    action_log: list[dict[str, Any]] = Field(default_factory=list)

    # --- deliberate choice of call path ------------------------------------ #

    def supports_function_calling(self) -> bool:
        """Return False to take the text/ReAct path.

        CrewAI only takes the native function-calling path when this returns
        truthy. We choose ReAct because it is the path the brief's two
        documented pitfalls concern, so handling it here is the stronger
        demonstration -- and it keeps the mock free of any provider-shaped
        tool-call JSON.
        """
        return False

    def supports_stop_words(self) -> bool:
        return False

    # --- the one abstract method ------------------------------------------- #

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        self.calls += 1
        msgs = _as_dicts(messages)
        agent_role = getattr(from_agent, "role", "(unknown agent)")
        agent_tools = list(getattr(from_agent, "tools", None) or [])
        task_text = _current_task_text(msgs)

        # Has a tool already produced a result for this agent on this task?
        # Read ONLY our own generated text -- see pitfall 1.
        observation = _observation_from_own_output(msgs)
        if observation is not None:
            return self._finish(agent_role, observation)

        if agent_tools:
            tool = agent_tools[0]
            try:
                args = _arguments_from_schema(tool, task_text)
            except ToolArgumentsUnavailable as exc:
                # Expected outcome, not an error: this task does not carry what
                # the tool needs, so the tool is simply not called. Recorded so
                # the transcript can show the tool was NOT invoked.
                self.action_log.append(
                    {
                        "agent": agent_role,
                        "tool": None,
                        "arguments": None,
                        "dispatched_by": "declared_schema",
                        "skipped_reason": str(exc),
                    }
                )
                return self._finish(
                    agent_role,
                    json.dumps(
                        {
                            "record_id": None,
                            "note": "no ORD-#### id present in this task; "
                            "check_order_status was not called",
                        }
                    ),
                    task_text=task_text,
                )
            tool_name = getattr(tool, "name", "unknown_tool")
            self.action_log.append(
                {
                    "agent": agent_role,
                    "tool": tool_name,
                    "arguments": args,
                    "dispatched_by": "declared_schema",
                    "schema_required_keys": _schema_required_keys(tool),
                }
            )
            return (
                f"Thought: I need to use the {tool_name} tool to answer this.\n"
                f"Action: {tool_name}\n"
                f"Action Input: {json.dumps(args)}\n"
            )

        # No tools wired to this agent -> it is the composer. It needs the full
        # prompt, because the context tasks' outputs are injected below the
        # "Current Task:" line in the same user message.
        return self._finish(agent_role, None, task_text=_full_user_text(msgs))

    # --- response construction --------------------------------------------- #

    def _finish(self, agent_role: str, observation: str | None, task_text: str = "") -> str:
        if observation is None:
            body = self._compose(task_text)
        else:
            body = observation
        return f"Thought: I now know the final answer\nFinal Answer: {body}"

    def _compose(self, task_text: str) -> str:
        """Composer agent: merge prior agents' JSON into the response schema."""
        blobs = _extract_json_objects(task_text)

        retrieval: dict[str, Any] = {}
        order: dict[str, Any] = {}
        for b in blobs:
            if "top_similarity" in b or "answered" in b:
                retrieval = b
            elif "escalation_score" in b or "record_id" in b:
                order = b

        answer = str(retrieval.get("answer") or "").strip()
        if not answer:
            answer = "No policy context was retrieved for this question."

        needs_human = bool(order.get("recommend_escalation")) or not bool(
            retrieval.get("answered", False)
        )
        reasons: list[str] = []
        if not retrieval.get("answered", False):
            reasons.append("the knowledge base does not cover this question")
        if order.get("recommend_escalation"):
            reasons.append("the order's escalation score is at or above the policy threshold")
        if not reasons:
            reasons.append("the retrieved policy context answers the question")
        reason = "; ".join(reasons)

        composed = {
            "answer": answer,
            "grounded": bool(retrieval.get("answered", False)),
            "sources": list(retrieval.get("sources") or []),
            "top_similarity": float(retrieval.get("top_similarity") or 0.0),
            "order_record_id": order.get("record_id"),
            "order_status": order.get("status"),
            "escalation_score": order.get("escalation_score"),
            "needs_human": needs_human,
            "reason": reason,
        }
        return json.dumps(composed)
