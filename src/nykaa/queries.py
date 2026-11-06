"""Shared query sets and relevance judgments.

Task 5 requires precision/recall on "the same >=5 queries from Task 4", and
Task 13 builds a 15-query evaluation set touching every KB topic. Centralising
the queries and their gold document sets here keeps those numbers comparable
and stops the two scripts drifting apart.

``gold_doc_ids`` is the set of documents that actually *answer* the query -- the
minimal defensible relevance judgment. Every in-scope query maps to exactly one
document, which keeps precision/recall arithmetic unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    text: str
    gold_doc_ids: frozenset[str]
    topic: str


# --------------------------------------------------------------------------- #
# In-scope queries (Task 4 needs >=3; Task 5 needs >=5)
# --------------------------------------------------------------------------- #

IN_SCOPE: list[Query] = [
    Query(
        text="How long do I have to return a pair of shoes?",
        gold_doc_ids=frozenset({"01_return_window"}),
        topic="return window by product category",
    ),
    Query(
        text="When will I get my money back for a cash on delivery order?",
        gold_doc_ids=frozenset({"02_cod_refund_timelines"}),
        topic="COD refund timelines",
    ),
    Query(
        text="How many days does standard delivery take to a metro pin code?",
        gold_doc_ids=frozenset({"03_delivery_slas"}),
        topic="delivery SLAs",
    ),
    Query(
        text="What is the warranty on a laptop I bought?",
        gold_doc_ids=frozenset({"05_warranty_terms"}),
        topic="warranty terms by category",
    ),
    Query(
        text="How many loyalty points do I need to redeem for a discount?",
        gold_doc_ids=frozenset({"07_loyalty_points"}),
        topic="loyalty-points redemption policy",
    ),
    Query(
        text="When does a support conversation get escalated to a human?",
        gold_doc_ids=frozenset({"12_escalation_matrix"}),
        topic="customer-support escalation matrix",
    ),
]


# --------------------------------------------------------------------------- #
# Deliberately out-of-scope queries -- these MUST trigger the fallback
# --------------------------------------------------------------------------- #

OUT_OF_SCOPE: list[str] = [
    "How do I install the latest NVIDIA CUDA driver on Ubuntu?",
    "What is the capital of Australia?",
    "Can you write me a Python function that sorts a list of integers?",
    "Who won the 2019 Cricket World Cup final?",
]


TOP_K = 3
