"""Environment verification: confirm the whole stack imports and is CPU-only.

Run with::

    uv run python scripts/00_verify_env.py

Exits non-zero if any required component is missing. This is the first thing a
grader should run after `uv sync`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Suppress telemetry BEFORE importing crewai / chromadb (silent network calls).
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

FAILURES: list[str] = []


def check(label: str, fn) -> None:
    try:
        detail = fn()
        print(f"  [ OK ] {label:<42} {detail}")
    except Exception as exc:  # noqa: BLE001 - we want the message, whatever it is
        FAILURES.append(f"{label}: {exc!r}")
        print(f"  [FAIL] {label:<42} {exc!r}")


def version(modname: str):
    def _inner():
        mod = __import__(modname)
        return getattr(mod, "__version__", "(no __version__)")

    return _inner


def main() -> int:
    print("=" * 72)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 72)
    print(f"python executable : {sys.executable}")
    print(f"python version    : {sys.version.split()[0]}")
    print(f"cwd               : {Path.cwd()}")
    print()

    print("Core libraries:")
    check("pydantic", version("pydantic"))
    check("fastapi", version("fastapi"))
    check("uvicorn", version("uvicorn"))
    check("chromadb", version("chromadb"))
    check("sentence_transformers", version("sentence_transformers"))
    check("torch", version("torch"))
    check("crewai", version("crewai"))
    check("langchain_core", version("langchain_core"))
    check("autogen_agentchat", version("autogen_agentchat"))

    print()
    print("Deep imports actually used by the code:")

    def _crewai_base():
        from crewai.llms.base_llm import BaseLLM  # noqa: F401

        return "BaseLLM reachable (MOCK_LLM extension point)"

    def _autogen_team():
        from autogen_agentchat.teams import RoundRobinGroupChat  # noqa: F401
        from autogen_agentchat.conditions import MaxMessageTermination  # noqa: F401

        return "RoundRobinGroupChat + MaxMessageTermination reachable"

    def _autogen_struct():
        from autogen_agentchat.messages import StructuredMessage  # noqa: F401

        return "StructuredMessage reachable (Task 14 verdict model)"

    def _lc_memory():
        from langchain_core.runnables.history import RunnableWithMessageHistory  # noqa: F401
        from langchain_core.chat_history import InMemoryChatMessageHistory  # noqa: F401

        return "RunnableWithMessageHistory + InMemoryChatMessageHistory"

    def _ws():
        from fastapi import FastAPI, WebSocket  # noqa: F401

        return "FastAPI WebSocket support present"

    check("crewai.llms.base_llm.BaseLLM", _crewai_base)
    check("autogen RoundRobinGroupChat", _autogen_team)
    check("autogen StructuredMessage", _autogen_struct)
    check("langchain session memory", _lc_memory)
    check("fastapi websocket", _ws)

    print()
    print("CPU-only / no-CUDA checks:")

    def _torch_cpu():
        import torch

        build = getattr(torch.version, "cuda", None)
        if build is not None:
            raise RuntimeError(f"torch was built WITH CUDA ({build}) - want CPU build")
        return f"torch {torch.__version__}, cuda build = {build!r}, device = cpu"

    def _no_nvidia():
        import importlib.util

        found = [m for m in ("nvidia", "cuda", "triton") if importlib.util.find_spec(m)]
        if found:
            raise RuntimeError(f"CUDA/GPU packages present: {found}")
        return "no nvidia-* / cuda-* / triton packages installed"

    check("torch is a CPU build", _torch_cpu)
    check("no CUDA packages installed", _no_nvidia)

    print()
    print("Network-suppression env vars:")
    for var in (
        "CREWAI_DISABLE_TELEMETRY",
        "OTEL_SDK_DISABLED",
        "ANONYMIZED_TELEMETRY",
    ):
        print(f"  {var} = {os.environ.get(var)!r}")

    print()
    print("=" * 72)
    if FAILURES:
        print(f"RESULT: FAILED - {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 72)
        return 1
    print("RESULT: PASS - full stack installed, imports clean, CPU-only, telemetry off.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
