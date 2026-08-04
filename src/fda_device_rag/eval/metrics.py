import math

WILSON_Z_95 = 1.96


def wilson_interval(hits: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, returned as
    (lower_pct, upper_pct) percentages in [0, 100]. Used instead of a
    normal-approximation interval because it stays well-behaved at small n
    and at p near 0 or 1 -- both regimes this project's per-stratum
    (n=12-13) Hit Rate@5 figures actually fall into.
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = max(0.0, center - margin) * 100
    hi = min(1.0, center + margin) * 100
    return (lo, hi)


def rank_of_first_gold_hit(ranked_ids: list[str], gold_ids: list[str]) -> int | None:
    """1-indexed rank of the first entry in ranked_ids that appears in
    gold_ids, or None if no entry does. When gold_ids has multiple chunk
    ids (a multi-chunk gold set), this returns the best (lowest) rank among
    them -- standard MRR convention, matching Hit Rate@5's own "any match
    counts" semantics rather than averaging across all gold chunks found.
    """
    gold_set = set(gold_ids)
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in gold_set:
            return rank
    return None


def hit_at_k(rank: int | None, k: int = 5) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0
