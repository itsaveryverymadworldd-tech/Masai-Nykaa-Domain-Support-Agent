"""Optional real-LLM backend, behind an environment flag.

The brief permits this explicitly: *"A real LLM API may optionally be wired in
behind an environment-variable flag, but every acceptance criterion below must be
demonstrably satisfied under MOCK_LLM alone."*

So this module is strictly additive:

* **Default is ``mock``.** Nothing changes unless you opt in.
* Every acceptance criterion still holds with ``NYKAA_LLM_BACKEND`` unset.
* Real mode is the only path in this project that touches the network, which is
  why it has to be asked for by name.

Configuration (all optional except the first)
---------------------------------------------
===============================  ==========================================
``NYKAA_LLM_BACKEND``            ``mock`` (default) or ``real``
``NYKAA_LLM_MODEL``              litellm model string, e.g. ``openai/gpt-4o-mini``,
                                 ``groq/llama-3.3-70b-versatile``,
                                 ``ollama/qwen2.5:3b``
``NYKAA_LLM_BASE_URL``           for OpenAI-compatible servers, e.g.
                                 ``http://localhost:11434/v1`` (Ollama)
``NYKAA_LLM_API_KEY_ENV``        name of the env var holding the key
                                 (default: ``OPENAI_API_KEY``)
``NYKAA_LLM_TEMPERATURE``        default ``0.0``, for reproducible output
===============================  ==========================================

Example::

    export NYKAA_LLM_BACKEND=real
    export NYKAA_LLM_MODEL=groq/llama-3.3-70b-versatile
    export NYKAA_LLM_API_KEY_ENV=GROQ_API_KEY
    export GROQ_API_KEY=...            # your own shell, never in the repo
    uv run python scripts/12_serve.py
"""

from __future__ import annotations

import os
from typing import Any

from nykaa import config

# --------------------------------------------------------------------------- #
# Mode
# --------------------------------------------------------------------------- #

_MOCK_VALUES = {"", "mock", "0", "false", "off", "none"}


def is_real() -> bool:
    """True when a real LLM backend was explicitly requested."""
    return str(getattr(config, "LLM_BACKEND", "mock")).strip().lower() not in _MOCK_VALUES


def backend_name() -> str:
    return "real" if is_real() else "mock"


def api_key_env() -> str:
    return os.environ.get("NYKAA_LLM_API_KEY_ENV", "OPENAI_API_KEY")


def resolve_api_key() -> str | None:
    """Read the key from the environment. Never logged, never written to disk."""
    value = os.environ.get(api_key_env(), "")
    return value or None


def base_url() -> str | None:
    return os.environ.get("NYKAA_LLM_BASE_URL") or None


def model_name() -> str:
    return os.environ.get("NYKAA_LLM_MODEL", "openai/gpt-4o-mini")


def temperature() -> float:
    try:
        return float(os.environ.get("NYKAA_LLM_TEMPERATURE", "0.0"))
    except ValueError:
        return 0.0


class BackendMisconfigured(RuntimeError):
    """Real mode was requested but cannot be satisfied."""


def validate() -> None:
    """Fail early and clearly if real mode is requested but unusable.

    A silent fallback to MOCK_LLM would be the worst outcome: the run would look
    successful while quietly not exercising the real model at all.
    """
    if not is_real():
        return

    url = base_url()
    key = resolve_api_key()

    # A local OpenAI-compatible server (Ollama, llama.cpp, vLLM, LM Studio)
    # legitimately needs no API key.
    local = bool(url) and any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    if not key and not local:
        raise BackendMisconfigured(
            f"NYKAA_LLM_BACKEND=real but no API key found. Set {api_key_env()} in "
            "your own shell (not in the repo), or point NYKAA_LLM_BASE_URL at a "
            "local OpenAI-compatible server. Refusing to silently fall back to "
            "MOCK_LLM, because that would look like success while testing nothing."
        )

    if "/" not in model_name() and not url:
        raise BackendMisconfigured(
            f"NYKAA_LLM_MODEL={model_name()!r} has no provider prefix. Use a litellm "
            "form such as 'openai/gpt-4o-mini' or 'groq/llama-3.3-70b-versatile'."
        )


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #


def build_crew_llm() -> Any:
    """The LLM the CrewAI agents use.

    Mock -> :class:`nykaa.mock_llm.MockLLM` (deterministic, offline, ReAct path).
    Real -> ``crewai.LLM``, which is litellm-backed and *does* advertise native
    function calling, so CrewAI takes its native tool-calling path instead of
    ReAct. That is a real behavioural difference worth watching for.
    """
    if not is_real():
        from nykaa.mock_llm import MockLLM

        return MockLLM(model="mock-llm")

    validate()
    from crewai import LLM

    kwargs: dict[str, Any] = {
        "model": model_name(),
        "temperature": temperature(),
    }
    url = base_url()
    if url:
        kwargs["base_url"] = url
    key = resolve_api_key()
    if key:
        kwargs["api_key"] = key
    return LLM(**kwargs)


def build_autogen_client() -> Any:
    """The model client the Autogen review team uses."""
    if not is_real():
        from nykaa.review import MockReviewClient

        return MockReviewClient()

    validate()
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    # Autogen takes a bare model name, not a litellm "provider/model" string.
    model = model_name().split("/", 1)[1] if "/" in model_name() else model_name()
    kwargs: dict[str, Any] = {"model": model}
    url = base_url()
    if url:
        kwargs["base_url"] = url
    key = resolve_api_key()
    if key:
        kwargs["api_key"] = key
    elif url:
        # Local servers usually ignore the key but the client insists on one.
        kwargs["api_key"] = "not-needed"
    return OpenAIChatCompletionClient(**kwargs)


def describe() -> list[str]:
    """Human-readable backend summary for banners and logs."""
    if not is_real():
        return [
            "LLM backend : MOCK_LLM (offline, no API keys, no network)",
        ]
    key_name = api_key_env()
    lines = [
        "LLM backend : REAL  (network calls enabled)",
        f"model       : {model_name()}",
        f"temperature : {temperature()}",
        f"api key     : {key_name} = {'set' if resolve_api_key() else 'unset'}",
    ]
    if base_url():
        lines.append(f"base url    : {base_url()}")
    lines.append(
        "NOTE        : this is an optional mode. Graded behaviour and every "
        "acceptance criterion hold under MOCK_LLM alone."
    )
    return lines
