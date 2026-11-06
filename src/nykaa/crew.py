"""The CrewAI crew (Part 2, Task 7).

Three agents, sequential process:

1. **Policy Retrieval Specialist** -- wired to ``rag_lookup`` only.
2. **Order Lookup Specialist**   -- wired to ``check_order_status`` only.
3. **Response Composer**         -- wired to no tools; merges 1 and 2 into one
   answer conforming to :class:`nykaa.schemas.SupportResponse`.

Part 4's Application-layer least-autonomy rule is enforced structurally here:
``check_order_status`` is constructed and attached to the Lookup Agent and to
nothing else. No other agent in the crew can reach it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nykaa import config
from nykaa.mock_llm import MockLLM
from nykaa.tools import build_check_order_status_tool, build_rag_lookup_tool


@dataclass
class CrewRun:
    """Everything one ``kickoff()`` produced, for the transcripts."""

    query: str
    record_id: str | None
    result: Any
    action_log: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0

    @property
    def tools_invoked(self) -> list[str]:
        return [a["tool"] for a in self.action_log if a.get("tool")]


def build_agents(llm: MockLLM):
    """Construct the three agents. Returns ``(retrieval, lookup, composer)``."""
    from crewai import Agent

    rag_tool = build_rag_lookup_tool()
    status_tool = build_check_order_status_tool()

    retrieval_agent = Agent(
        role="Policy Retrieval Specialist",
        goal=(
            "Answer Nykaa policy questions using ONLY the knowledge base, via the "
            "rag_lookup tool."
        ),
        backstory=(
            "You are the team's policy expert. You never answer from memory -- you "
            "always call rag_lookup and report what the knowledge base says. If the "
            "knowledge base does not cover the question, you say so plainly."
        ),
        tools=[rag_tool],
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )

    lookup_agent = Agent(
        role="Order Lookup Specialist",
        goal="Look up a single Nykaa order by id and report its status and escalation score.",
        backstory=(
            "You are the only member of this crew permitted to query order records. "
            "You call check_order_status exactly once for the order you are given and "
            "report the record faithfully, including the escalation recommendation."
        ),
        # Least autonomy: check_order_status is wired here and NOWHERE else.
        tools=[status_tool],
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )

    composer_agent = Agent(
        role="Response Composer",
        goal="Combine the policy answer and the order lookup into one final customer-facing reply.",
        backstory=(
            "You write the final reply a customer would read. You merge what the "
            "retrieval and lookup specialists found, you never invent facts, and you "
            "escalate to a human whenever the evidence is weak or the order is at risk."
        ),
        tools=[],  # deliberately none: the composer must not call tools itself
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )

    return retrieval_agent, lookup_agent, composer_agent


def run_crew(query: str, record_id: str | None = None) -> CrewRun:
    """Run one full crew pass for ``query`` (and optional ``record_id``)."""
    from crewai import Crew, Process, Task

    from nykaa.schemas import SupportResponse

    llm = MockLLM(model="mock-llm")
    retrieval_agent, lookup_agent, composer_agent = build_agents(llm)

    retrieval_task = Task(
        description=(
            "Answer this Nykaa customer policy question using the rag_lookup tool. "
            f"Question: {query}"
        ),
        expected_output="A grounded policy answer with its source document ids.",
        agent=retrieval_agent,
    )

    if record_id:
        lookup_task = Task(
            description=(
                "Look up the Nykaa order with record_id "
                f"{record_id} using the check_order_status tool, and report its "
                "status, value and escalation score."
            ),
            expected_output="The order record with its escalation recommendation.",
            agent=lookup_agent,
        )
    else:
        lookup_task = Task(
            description=(
                "No order record id was supplied for this conversation, so there is "
                "nothing to look up. Report that no order lookup was required."
            ),
            expected_output="A short statement that no order lookup was needed.",
            agent=lookup_agent,
        )

    composer_task = Task(
        description=(
            "Combine the policy answer and the order lookup above into ONE final "
            "customer-facing reply. Use only what the specialist agents reported. "
            "Set needs_human to true if the policy question was not covered by the "
            "knowledge base or if the order's escalation score recommends escalation."
        ),
        expected_output="A single JSON object conforming to the SupportResponse schema.",
        agent=composer_agent,
        context=[retrieval_task, lookup_task],
        output_pydantic=SupportResponse,  # crewai 1.x name for response_format
    )

    crew = Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[retrieval_task, lookup_task, composer_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return CrewRun(
        query=query,
        record_id=record_id,
        result=result,
        action_log=list(llm.action_log),
        llm_calls=llm.calls,
    )


def assert_least_autonomy() -> dict[str, Any]:
    """Evidence for the Part 4 least-autonomy guard.

    Asserts that ``check_order_status`` is reachable from exactly one agent.
    """
    llm = MockLLM(model="mock-llm")
    retrieval_agent, lookup_agent, composer_agent = build_agents(llm)
    holders: list[str] = []
    for agent in (retrieval_agent, lookup_agent, composer_agent):
        for t in getattr(agent, "tools", None) or []:
            if getattr(t, "name", "") == "check_order_status":
                holders.append(agent.role)
    return {
        "agents_total": 3,
        "check_order_status_holders": holders,
        "holder_count": len(holders),
        "enforced": holders == ["Order Lookup Specialist"],
    }
