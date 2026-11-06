"""Tools for the Nykaa support crew (Part 2, Tasks 6-7).

Two tools, kept deliberately separate:

* ``check_order_status`` -- structured record lookup + escalation score (Task 6)
* ``rag_lookup``         -- grounded policy retrieval from the KB (Task 3-5)

Part 4's least-autonomy rule depends on this separation: only the Lookup Agent
is ever wired to ``check_order_status``.

The pure functions are separated from the CrewAI ``@tool`` wrappers so they can
be unit-tested and demonstrated without spinning up a crew.
"""

from __future__ import annotations

import json
from typing import Any

from nykaa import config

# --------------------------------------------------------------------------- #
# Task 6 -- escalation score
# --------------------------------------------------------------------------- #

#: Value at which the order-value component saturates. Chosen as the top of the
#: Electronics price band in dataset.py, so the most expensive order in the
#: catalogue reaches 1.0 and everything else scales below it.
VALUE_SATURATION_INR = 25_000

#: Weights for the three signal components. Delayed shipment dominates because
#: it is the single strongest predictor of a customer contacting support;
#: recency and value split the remainder.
W_DELAY = 0.40
W_RECENCY = 0.30
W_VALUE = 0.30

#: Multiplier reflecting how much a given status still needs human attention.
#: A Delivered or Refunded order is largely resolved; a Placed, Shipped or
#: Returned order may still need intervention.
STATUS_URGENCY: dict[str, float] = {
    "Placed": 1.00,
    "Shipped": 1.00,
    "Returned": 0.85,
    "Delivered": 0.55,
    "Refunded": 0.35,
}

#: Days used to normalise the recency signal (max in dataset.py is 30).
RECENCY_HORIZON_DAYS = 30


def normalize_recency(days_since_created: int) -> float:
    """Normalised recency signal in [0, 1] derived from ``days_since_created``.

    A 30-day-old order scores 1.0 (stale, most urgent); a brand-new order
    scores 0.0.
    """
    clamped = max(0, min(RECENCY_HORIZON_DAYS, days_since_created))
    return clamped / RECENCY_HORIZON_DAYS


def compute_escalation_score(record: dict[str, Any]) -> float:
    """Designed escalation score in [0, 1] -- a weighted blend, not a boolean.

    ``score = status_urgency * (0.40*delayed + 0.30*recency + 0.30*value)``

    The brief requires this to combine the order's fields with a normalised
    recency signal rather than being "a bare boolean OR", which is why all
    three components are continuous and independently weighted.
    """
    delay_c = 1.0 if record["delayed_shipment"] else 0.0
    recency_c = normalize_recency(int(record["days_since_created"]))
    value_c = min(1.0, float(record["order_value_inr"]) / VALUE_SATURATION_INR)

    raw = W_DELAY * delay_c + W_RECENCY * recency_c + W_VALUE * value_c
    urgency = STATUS_URGENCY.get(record["status"], 1.0)
    return round(max(0.0, min(1.0, raw * urgency)), 6)


def escalation_breakdown(record: dict[str, Any]) -> dict[str, Any]:
    """Human-readable decomposition of the score, for transcripts."""
    delay_c = 1.0 if record["delayed_shipment"] else 0.0
    recency_c = normalize_recency(int(record["days_since_created"]))
    value_c = min(1.0, float(record["order_value_inr"]) / VALUE_SATURATION_INR)
    urgency = STATUS_URGENCY.get(record["status"], 1.0)
    return {
        "delay_component": delay_c,
        "recency_component": round(recency_c, 6),
        "value_component": round(value_c, 6),
        "status_urgency": urgency,
        "weighted_sum": round(
            W_DELAY * delay_c + W_RECENCY * recency_c + W_VALUE * value_c, 6
        ),
        "escalation_score": compute_escalation_score(record),
    }


# --------------------------------------------------------------------------- #
# Task 6 -- record lookup
# --------------------------------------------------------------------------- #


def _orders_index() -> dict[str, dict[str, Any]]:
    from dataset import ORDERS

    return {o["record_id"]: o for o in ORDERS}


def check_order_status(record_id: str) -> dict[str, Any]:
    """Look up one order and return its status, value and escalation score.

    Returns an ``{"error": ...}`` dict for an unknown id rather than raising, so
    an agent cannot crash the crew by hallucinating a record id.
    """
    orders = _orders_index()
    record = orders.get(record_id.strip().upper())
    if record is None:
        # Be forgiving about case and whitespace, but not about existence.
        for rid, rec in orders.items():
            if rid.lower() == record_id.strip().lower():
                record = rec
                break
    if record is None:
        return {
            "error": "record_not_found",
            "record_id": record_id,
            "message": f"No order with record_id {record_id!r} exists in the dataset.",
        }
    return {
        "record_id": record["record_id"],
        "category": record["category"],
        "status": record["status"],
        "order_value_inr": record["order_value_inr"],
        "days_since_created": record["days_since_created"],
        "delayed_shipment": record["delayed_shipment"],
        "escalation_score": compute_escalation_score(record),
        "recommend_escalation": compute_escalation_score(record)
        >= config.DEFAULT_ESCALATION_THRESHOLD,
    }


# --------------------------------------------------------------------------- #
# CrewAI tool wrappers
# --------------------------------------------------------------------------- #


def build_rag_lookup_tool():
    """CrewAI tool: grounded policy retrieval over the deployed collection."""
    from crewai.tools import tool

    from nykaa.rag import get_client, grounded_answer, retrieve

    @tool("rag_lookup")
    def rag_lookup(query: str) -> str:
        """Search the Nykaa policy knowledge base and return a grounded answer.

        Use this for any question about policy: return windows, refunds,
        delivery times, reverse pickup, warranty, cancellation, loyalty points,
        payment failures, size exchanges, damaged items, international shipping
        or escalation.

        Args:
            query: The policy question to look up, in natural language.
        """
        collection = get_client().get_or_create_collection(
            name=config.COLLECTION_SENTENCE
        )
        result = grounded_answer(query, collection)
        # De-duplicate source ids, preserving rank order: the same document can
        # contribute several retrieved chunks.
        sources: list[str] = []
        for h in result.hits:
            if h.doc_id not in sources:
                sources.append(h.doc_id)
        return json.dumps(
            {
                "answered": result.answered,
                "answer": result.answer,
                "top_similarity": result.top_similarity,
                "threshold": result.threshold,
                "sources": sources,
            },
            indent=2,
        )

    return rag_lookup


def build_check_order_status_tool():
    """CrewAI tool: order record lookup + escalation score.

    Wire this to the Lookup Agent ONLY (Part 4, Application-layer least autonomy).
    """
    from crewai.tools import tool

    @tool("check_order_status")
    def check_order_status_tool(record_id: str) -> str:
        """Look up a Nykaa order by its record id.

        Returns the order's status, category, value in INR, days since it was
        created, whether the shipment is delayed, and a designed escalation
        score in [0, 1] with a recommendation on whether to escalate to a human.

        Args:
            record_id: The order record id, e.g. "ORD-0007".
        """
        return json.dumps(check_order_status(record_id), indent=2)

    return check_order_status_tool
