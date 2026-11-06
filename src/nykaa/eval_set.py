"""The 15-query evaluation set (Part 3, Task 13).

Requirement: 15 queries, at least one touching **every** required KB topic, plus
at least 2 deliberately out-of-scope or edge-case queries.

Composition
    12 in-scope queries, one per required knowledge-base topic
     1 order-status query exercising the record lookup tool
     2 deliberately out-of-scope queries that must be refused
    ---
    15 total

``expected_facts`` are literal phrases drawn from the KB documents. Because
generation is extractive, a correct answer must contain them; their absence means
the right document was not retrieved or was not used.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    topic: str
    kind: str  # "in_scope" | "order" | "out_of_scope"
    expected_doc_ids: frozenset[str] = frozenset()
    expected_facts: tuple[str, ...] = field(default_factory=tuple)
    record_id: str | None = None
    expect_refusal: bool = False


EVAL_SET: list[EvalCase] = [
    # ---- one per required KB topic (12) ---------------------------------- #
    EvalCase(
        case_id="T01",
        query="How long do I have to return a pair of shoes?",
        topic="return window by product category",
        kind="in_scope",
        expected_doc_ids=frozenset({"01_return_window"}),
        expected_facts=("7 days of delivery",),
    ),
    EvalCase(
        case_id="T02",
        query="When will I get my money back for a cash on delivery order?",
        topic="COD refund timelines",
        kind="in_scope",
        expected_doc_ids=frozenset({"02_cod_refund_timelines"}),
        expected_facts=("3 to 5 working days",),
    ),
    EvalCase(
        case_id="T03",
        query="How many days does standard delivery take to a metro pin code?",
        topic="delivery SLAs",
        kind="in_scope",
        expected_doc_ids=frozenset({"03_delivery_slas"}),
        expected_facts=("3 to 5 working days",),
    ),
    EvalCase(
        case_id="T04",
        query="Can a return be picked up from my home address?",
        topic="reverse-pickup eligibility",
        kind="in_scope",
        expected_doc_ids=frozenset({"04_reverse_pickup_eligibility"}),
        expected_facts=("19,000 pin codes",),
    ),
    EvalCase(
        case_id="T05",
        query="What is the warranty on a laptop I bought?",
        topic="warranty terms by category",
        kind="in_scope",
        expected_doc_ids=frozenset({"05_warranty_terms"}),
        expected_facts=("12 months",),
    ),
    EvalCase(
        case_id="T06",
        query="Can I cancel an order that has already been shipped?",
        topic="order-cancellation policy",
        kind="in_scope",
        expected_doc_ids=frozenset({"06_order_cancellation"}),
        expected_facts=("no longer possible",),
    ),
    EvalCase(
        case_id="T07",
        query="How many loyalty points do I need to redeem for a discount?",
        topic="loyalty-points redemption policy",
        kind="in_scope",
        expected_doc_ids=frozenset({"07_loyalty_points"}),
        expected_facts=("multiples of 100",),
    ),
    EvalCase(
        case_id="T08",
        query="My payment failed but money was deducted. What happens now?",
        topic="payment-failure/retry policy",
        kind="in_scope",
        expected_doc_ids=frozenset({"08_payment_failure_retry"}),
        expected_facts=("5 to 7 working days",),
    ),
    EvalCase(
        case_id="T09",
        query="Can I exchange a t-shirt for a bigger size?",
        topic="size-exchange policy",
        kind="in_scope",
        expected_doc_ids=frozenset({"09_size_exchange"}),
        expected_facts=("7 days of delivery",),
    ),
    EvalCase(
        case_id="T10",
        query="My order arrived damaged. What should I do?",
        topic="damaged-item claim process",
        kind="in_scope",
        expected_doc_ids=frozenset({"10_damaged_item_claims"}),
        expected_facts=("48 hours",),
    ),
    EvalCase(
        case_id="T11",
        query="Do you deliver to Dubai?",
        topic="international shipping restrictions",
        kind="in_scope",
        expected_doc_ids=frozenset({"11_international_shipping"}),
        expected_facts=("India only",),
    ),
    EvalCase(
        case_id="T12",
        query="When does my complaint get escalated to a human agent?",
        topic="customer-support escalation matrix",
        kind="in_scope",
        expected_doc_ids=frozenset({"12_escalation_matrix"}),
        expected_facts=("Tier 2",),
    ),
    # ---- order-status lookup (1) ---------------------------------------- #
    EvalCase(
        case_id="T13",
        query="What is the current status of my order ORD-0037?",
        topic="order record lookup",
        kind="order",
        record_id="ORD-0037",
        expected_facts=("ORD-0037",),
    ),
    # ---- deliberately out-of-scope / edge cases (2) --------------------- #
    EvalCase(
        case_id="T14",
        query="How do I install the latest NVIDIA CUDA driver on Ubuntu?",
        topic="out-of-scope: technical support outside the KB",
        kind="out_of_scope",
        expect_refusal=True,
    ),
    EvalCase(
        case_id="T15",
        query="Who won the 2019 Cricket World Cup final?",
        topic="out-of-scope: general knowledge",
        kind="out_of_scope",
        expect_refusal=True,
    ),
]


REQUIRED_TOPICS = {
    "return window by product category",
    "COD refund timelines",
    "delivery SLAs",
    "reverse-pickup eligibility",
    "warranty terms by category",
    "order-cancellation policy",
    "loyalty-points redemption policy",
    "payment-failure/retry policy",
    "size-exchange policy",
    "damaged-item claim process",
    "international shipping restrictions",
    "customer-support escalation matrix",
}
