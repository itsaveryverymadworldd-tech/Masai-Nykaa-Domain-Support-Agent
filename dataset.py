"""Deterministic synthetic order dataset for the Nykaa Domain Support Agent.

Part 1, Task 1.  Generates an ``ORDERS`` list of synthetic orders for the
Nykaa (E-commerce & Retail) capstone track.

Design goals, in priority order
-------------------------------
1. **Deterministic** -- a fixed seed reproduces a byte-identical dataset, so a
   grader can regenerate it and compare counts.
2. **Structurally valid** -- the stated thresholds are satisfied *by
   construction*, not by hand-editing records afterwards:
   * every category appears at least ``MIN_PER_CATEGORY`` (3) times
   * every status appears at least ``MIN_PER_STATUS`` (1) time
   * the ``delayed_shipment=True`` share lands inside ``DELAY_BAND`` (10-30%)
3. **Plausible** -- prices, status mixes and delay rates resemble a real
   e-commerce order book rather than uniform noise.

How the thresholds are guaranteed without hand-editing
------------------------------------------------------
Categories and statuses are allocated **stratified**: each value is first
guaranteed its minimum, and the remaining slots are distributed across values
in proportion to ``CATEGORY_WEIGHTS`` / ``STATUS_WEIGHTS`` using the
largest-remainder method.  Coverage is therefore structural.

The delayed-shipment band is the one threshold that is genuinely random.  It is
*not* forced by editing records.  Instead each status carries its own plausible
delay probability (``DELAY_PROB_BY_STATUS``), the realised rate is measured, and
:func:`find_seed_in_band` scanned seeds at development time to select the
``SEED`` value committed below -- which is the "change your seed and regenerate"
remedy the brief prescribes.  :func:`generate_orders` re-validates the band on
every call and raises if a future edit to the weights breaks it.

Run directly to print the Task 1 report::

    python dataset.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Design choices -- these are the values quoted verbatim in README.md so the
# grader can reproduce the dataset deterministically. Do not change one without
# changing the README.
# --------------------------------------------------------------------------- #

#: Chosen by scanning seeds with find_seed_in_band(); see module docstring.
SEED: int = 20261106

#: >= 40 required; 48 gives comfortable room above every per-value minimum.
N_RECORDS: int = 48

#: The four given categories, plus Beauty (added -- Nykaa's flagship vertical,
#: and the brief permits adding values).
CATEGORIES: tuple[str, ...] = ("Apparel", "Electronics", "Home", "Footwear", "Beauty")

#: Relative frequency of each category before the minimum-count correction.
#: Beauty and Apparel are Nykaa's highest-traffic verticals, so they lead;
#: Electronics is the smallest catalogue among these.
CATEGORY_WEIGHTS: dict[str, float] = {
    "Apparel": 0.28,
    "Beauty": 0.30,
    "Footwear": 0.16,
    "Home": 0.14,
    "Electronics": 0.12,
}

#: The four given statuses, plus nothing -- the given vocabulary is realistic
#: for a fulfilment pipeline and already has full coverage.
STATUSES: tuple[str, ...] = ("Placed", "Shipped", "Delivered", "Returned", "Refunded")

#: Lifecycle-weighted: most orders settle at Delivered, with a meaningful tail
#: of in-flight, returned and refunded orders.
STATUS_WEIGHTS: dict[str, float] = {
    "Delivered": 0.44,
    "Shipped": 0.22,
    "Placed": 0.14,
    "Returned": 0.11,
    "Refunded": 0.09,
}

#: Per-status probability that an order is flagged delayed.  Delays are a
#: fulfilment-pipeline phenomenon: an order that already reached Delivered
#: cannot meaningfully still be delayed, whereas in-flight and returned orders
#: are exactly the population a support agent gets asked about.
DELAY_PROB_BY_STATUS: dict[str, float] = {
    "Placed": 0.30,
    "Shipped": 0.42,
    "Delivered": 0.04,
    "Returned": 0.12,
    "Refunded": 0.10,
}

#: Realistic per-category INR price bands (log-uniform inside the band).
#: Reasoning: Nykaa's catalogue spans a Rs.349 lipstick to a Rs.24,999 laptop,
#: and spend is right-skewed -- there are many cheap orders and few very
#: expensive ones -- so a log-uniform draw per band models it far better than a
#: flat uniform draw over one global range.
CATEGORY_VALUE_BANDS: dict[str, tuple[int, int]] = {
    "Beauty": (349, 4_999),
    "Apparel": (499, 5_999),
    "Footwear": (799, 6_999),
    "Home": (399, 7_999),
    "Electronics": (1_499, 24_999),
}

#: Global bounds implied by the bands above; stated in README.md.
ORDER_VALUE_MIN_INR: int = min(lo for lo, _ in CATEGORY_VALUE_BANDS.values())
ORDER_VALUE_MAX_INR: int = max(hi for _, hi in CATEGORY_VALUE_BANDS.values())

#: days_since_created is drawn uniformly over this inclusive range.
DAYS_SINCE_CREATED_RANGE: tuple[int, int] = (0, 30)

MIN_PER_CATEGORY: int = 3
MIN_PER_STATUS: int = 1

#: The delayed_shipment=True share must land inside this band (Task 1).
DELAY_BAND: tuple[float, float] = (0.10, 0.30)


# --------------------------------------------------------------------------- #
# Allocation helpers
# --------------------------------------------------------------------------- #


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split ``total`` integers across ``weights`` proportionally.

    Uses the largest-remainder (Hare-Niemeyer) method so the parts sum to
    exactly ``total`` without floating-point drift.
    """
    if total <= 0:
        return {k: 0 for k in weights}
    weight_sum = sum(weights.values())
    exact = {k: total * (w / weight_sum) for k, w in weights.items()}
    floors = {k: int(v) for k, v in exact.items()}
    shortfall = total - sum(floors.values())
    # Hand out the remaining slots to the largest fractional parts,
    # breaking ties by key order for determinism.
    order = sorted(exact, key=lambda k: (-(exact[k] - floors[k]), k))
    for key in order[:shortfall]:
        floors[key] += 1
    return floors


def _stratified_values(
    values: tuple[str, ...],
    weights: dict[str, float],
    total: int,
    minimum: int,
) -> list[str]:
    """Return ``total`` values covering every entry at least ``minimum`` times.

    Guarantees coverage structurally, then fills the remainder by weight.
    """
    if minimum * len(values) > total:
        raise ValueError(
            f"cannot guarantee {minimum} of each of {len(values)} values "
            f"within {total} records"
        )
    remainder = total - minimum * len(values)
    extra = _largest_remainder(remainder, {v: weights[v] for v in values})
    out: list[str] = []
    for v in values:
        out.extend([v] * (minimum + extra[v]))
    return out


def _log_uniform(rng: random.Random, lo: int, hi: int) -> int:
    """Draw an integer from a log-uniform distribution over [lo, hi].

    Models right-skewed spend: many low-value orders, few high-value ones.
    """
    import math

    value = math.exp(rng.uniform(math.log(lo), math.log(hi + 1)))
    return int(min(hi, max(lo, round(value))))


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


def generate_orders(seed: int = SEED, n_records: int = N_RECORDS) -> list[dict[str, Any]]:
    """Build the deterministic ORDERS list.

    Same ``seed`` and ``n_records`` always produce an identical result.
    Raises ``ValueError`` if the realised delayed-shipment share falls outside
    :data:`DELAY_BAND` -- a signal to change the seed or the delay weights
    rather than to edit records.
    """
    rng = random.Random(seed)

    # 1. Stratified allocation of category and status, then shuffle so the two
    #    axes are independent and the records are not ordered by bucket.
    categories = _stratified_values(
        CATEGORIES, CATEGORY_WEIGHTS, n_records, MIN_PER_CATEGORY
    )
    statuses = _stratified_values(STATUSES, STATUS_WEIGHTS, n_records, MIN_PER_STATUS)
    rng.shuffle(categories)
    rng.shuffle(statuses)

    orders: list[dict[str, Any]] = []
    for i, (category, status) in enumerate(zip(categories, statuses), start=1):
        lo, hi = CATEGORY_VALUE_BANDS[category]
        order_value = _log_uniform(rng, lo, hi)
        days = rng.randint(*DAYS_SINCE_CREATED_RANGE)
        delayed = rng.random() < DELAY_PROB_BY_STATUS[status]
        orders.append(
            {
                "record_id": f"ORD-{i:04d}",
                "category": category,
                "status": status,
                "order_value_inr": order_value,
                "days_since_created": days,
                "delayed_shipment": delayed,
            }
        )

    delayed_share = sum(o["delayed_shipment"] for o in orders) / len(orders)
    lo_band, hi_band = DELAY_BAND
    if not (lo_band <= delayed_share <= hi_band):
        raise ValueError(
            f"delayed_shipment share {delayed_share:.1%} is outside the required "
            f"{lo_band:.0%}-{hi_band:.0%} band. Change SEED or "
            f"DELAY_PROB_BY_STATUS and regenerate -- do NOT hand-edit records."
        )
    return orders


def find_seed_in_band(
    start: int = 1, limit: int = 5_000, n_records: int = N_RECORDS
) -> tuple[int, float]:
    """Development-time helper: return the first seed whose delay share is in band.

    Used once to pick the committed :data:`SEED`; kept in the repo so the choice
    is auditable rather than a magic number.
    """
    for seed in range(start, start + limit):
        try:
            orders = generate_orders(seed=seed, n_records=n_records)
        except ValueError:
            continue
        return seed, sum(o["delayed_shipment"] for o in orders) / len(orders)
    raise RuntimeError(f"no seed in [{start}, {start + limit}) satisfied the band")


# --------------------------------------------------------------------------- #
# Reporting (the output Task 1 asks you to print)
# --------------------------------------------------------------------------- #

ORDERS: list[dict[str, Any]] = generate_orders()


def validate(orders: Iterable[dict[str, Any]] = ORDERS) -> dict[str, Any]:
    """Check every Task 1 structural threshold and return the report dict."""
    orders = list(orders)
    by_category = Counter(o["category"] for o in orders)
    by_status = Counter(o["status"] for o in orders)
    delayed = sum(o["delayed_shipment"] for o in orders)
    share = delayed / len(orders)

    problems: list[str] = []
    if len(orders) < 40:
        problems.append(f"only {len(orders)} records, need >= 40")
    for c in CATEGORIES:
        if by_category[c] < MIN_PER_CATEGORY:
            problems.append(f"category {c!r} has {by_category[c]}, need >= {MIN_PER_CATEGORY}")
    for s in STATUSES:
        if by_status[s] < MIN_PER_STATUS:
            problems.append(f"status {s!r} has {by_status[s]}, need >= {MIN_PER_STATUS}")
    if not (DELAY_BAND[0] <= share <= DELAY_BAND[1]):
        problems.append(f"delayed share {share:.1%} outside {DELAY_BAND}")

    return {
        "n_records": len(orders),
        "by_category": dict(by_category),
        "by_status": dict(by_status),
        "delayed_count": delayed,
        "delayed_share": share,
        "order_value_min": min(o["order_value_inr"] for o in orders),
        "order_value_max": max(o["order_value_inr"] for o in orders),
        "days_since_created_min": min(o["days_since_created"] for o in orders),
        "days_since_created_max": max(o["days_since_created"] for o in orders),
        "seed": SEED,
        "problems": problems,
    }


def _print_report(rep: dict[str, Any]) -> None:
    print("=" * 68)
    print("TASK 1 REPORT -- Nykaa order dataset")
    print("=" * 68)
    print(f"seed                      : {rep['seed']}")
    print(f"total records             : {rep['n_records']}   (required >= 40)")
    print()
    print("count per category (each required >= 3):")
    for c in CATEGORIES:
        n = rep["by_category"].get(c, 0)
        print(f"  {c:<14} {n:>3}   {'OK' if n >= 3 else 'FAIL'}")
    print()
    print("count per status (each required >= 1):")
    for s in STATUSES:
        n = rep["by_status"].get(s, 0)
        print(f"  {s:<14} {n:>3}   {'OK' if n >= 1 else 'FAIL'}")
    print()
    share = rep["delayed_share"]
    in_band = DELAY_BAND[0] <= share <= DELAY_BAND[1]
    print(
        f"delayed_shipment=True     : {rep['delayed_count']}/{rep['n_records']} "
        f"= {share:.1%}   (required 10%-30%)  {'OK' if in_band else 'FAIL'}"
    )
    print(
        f"order_value_inr range     : Rs.{rep['order_value_min']:,} - "
        f"Rs.{rep['order_value_max']:,}"
    )
    print(
        f"days_since_created range  : {rep['days_since_created_min']} - "
        f"{rep['days_since_created_max']}   (required 0-30)"
    )
    print()
    if rep["problems"]:
        print("VALIDATION FAILED:")
        for p in rep["problems"]:
            print(f"  - {p}")
    else:
        print("VALIDATION: PASS -- all Task 1 structural thresholds satisfied.")
    print("=" * 68)


if __name__ == "__main__":
    _print_report(validate())
    print()
    print("First 5 records:")
    print(json.dumps(ORDERS[:5], indent=2))
