"""Task 6 -- the escalation score and its justified threshold.

Computes ``check_order_status`` over the whole generated dataset, shows the
score distribution, and sets the escalation threshold from that distribution
rather than picking a round number.

Run with::

    uv run python scripts/06_escalation_score.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nykaa import config  # noqa: E402
from nykaa.tools import (  # noqa: E402
    STATUS_URGENCY,
    W_DELAY,
    W_RECENCY,
    W_VALUE,
    VALUE_SATURATION_INR,
    check_order_status,
    compute_escalation_score,
    escalation_breakdown,
)

from dataset import ORDERS  # noqa: E402


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (the common 'inclusive' definition)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def main() -> int:
    out: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    scores = [compute_escalation_score(r) for r in ORDERS]

    say("=" * 78)
    say("TASK 6 -- DESIGNED ESCALATION SCORE + JUSTIFIED THRESHOLD")
    say("=" * 78)
    say()
    say("FORMULA")
    say("-" * 78)
    say("  escalation_score = status_urgency * (w_delay*delayed + w_recency*recency + w_value*value)")
    say()
    say(f"    w_delay    = {W_DELAY}   delayed flag (1.0 if shipment delayed else 0.0)")
    say(f"    w_recency  = {W_RECENCY}   days_since_created normalised over 30 days")
    say(f"    w_value    = {W_VALUE}   order_value_inr saturating at Rs.{VALUE_SATURATION_INR:,}")
    say()
    say("  status_urgency multipliers:")
    for status, mult in sorted(STATUS_URGENCY.items()):
        say(f"      {status:<12} {mult}")
    say()
    say("  This is a weighted blend of three continuous components, not a")
    say("  boolean OR: two orders can both be delayed yet score differently")
    say("  depending on how stale and how valuable they are.")
    say()

    say("DISTRIBUTION OF escalation_score ACROSS THE GENERATED DATASET")
    say("-" * 78)
    for pct in (10, 25, 50, 75, 80, 90, 95):
        say(f"    p{pct:<3} {percentile(scores, pct):.4f}")
    say(f"    min  {min(scores):.4f}")
    say(f"    max  {max(scores):.4f}")
    say(f"    mean {statistics.fmean(scores):.4f}")
    say()

    threshold = round(percentile(scores, 80), 4)
    above = [s for s in scores if s >= threshold]
    say("THRESHOLD JUSTIFICATION")
    say("-" * 78)
    say(f"  Chosen threshold = p80 of escalation_score = {threshold:.4f}")
    say(f"  Orders at or above it: {len(above)}/{len(scores)} = "
        f"{100 * len(above) / len(scores):.1f}% of the dataset")
    days_above = [r["days_since_created"] for r in ORDERS
                  if compute_escalation_score(r) >= threshold]
    say(f"  Their days_since_created: median {statistics.median(days_above):.0f}, "
        f"IQR {percentile(days_above, 25):.0f}-{percentile(days_above, 75):.0f}")
    say("  i.e. the threshold corresponds to the stalest ~20% of orders, which")
    say("  is precisely the population a support team wants a human to see.")
    say()

    say("DISTRIBUTION COMPARISON")
    say("-" * 78)
    del_scores = [compute_escalation_score(r) for r in ORDERS if r["delayed_shipment"]]
    ok_scores = [compute_escalation_score(r) for r in ORDERS if not r["delayed_shipment"]]
    say(f"  delayed orders      : n={len(del_scores):>2}  mean score {statistics.fmean(del_scores):.4f}")
    say(f"  non-delayed orders  : n={len(ok_scores):>2}  mean score {statistics.fmean(ok_scores):.4f}")
    say(f"  delayed orders are on average "
        f"{statistics.fmean(del_scores) - statistics.fmean(ok_scores):+.4f} higher")
    say()

    say("WORKED EXAMPLES (check_order_status on real records)")
    say("-" * 78)
    ranked = sorted(ORDERS, key=compute_escalation_score, reverse=True)
    sample_ids = [ranked[0]["record_id"], ranked[1]["record_id"],
                  ranked[len(ranked) // 2]["record_id"], ranked[-1]["record_id"]]
    for rid in sample_ids:
        res = check_order_status(rid)
        bd = escalation_breakdown({**res, "order_value_inr": res["order_value_inr"]})
        say()
        say(f"  {res['record_id']}  ({res['category']}, {res['status']})")
        say(f"      value Rs.{res['order_value_inr']:,}  "
            f"days_since_created {res['days_since_created']}  "
            f"delayed={res['delayed_shipment']}")
        say(f"      delay_c={bd['delay_component']:.2f} "
            f"recency_c={bd['recency_component']:.4f} "
            f"value_c={bd['value_component']:.4f} "
            f"urgency={bd['status_urgency']:.2f}")
        say(f"      weighted_sum={bd['weighted_sum']:.4f}  "
            f"-> escalation_score={res['escalation_score']:.4f}")
        say(f"      recommend_escalation={res['recommend_escalation']}")
    say()

    say("UNKNOWN RECORD HANDLING (agents cannot crash the crew)")
    say("-" * 78)
    say(f"  check_order_status('ORD-9999') -> {check_order_status('ORD-9999')}")
    say()

    say("=" * 78)
    say(f">>> SET config.DEFAULT_ESCALATION_THRESHOLD = {threshold} <<<")
    say("=" * 78)

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRANSCRIPTS_DIR / "06_escalation_score.txt").write_text(
        "\n".join(out) + "\n", encoding="utf-8"
    )
    print("\n[transcript written to transcripts/06_escalation_score.txt]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
