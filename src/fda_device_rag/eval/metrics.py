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
