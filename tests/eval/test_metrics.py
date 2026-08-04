import math
import pytest

from fda_device_rag.eval.metrics import wilson_interval


def test_wilson_interval_matches_reference_values():
    cases = [
        ((38, 50), (62.5871, 85.7027)),
        ((0, 50), (0.0000, 7.1350)),
        ((50, 50), (92.8650, 100.0000)),
        ((25, 50), (36.6443, 63.3557)),
        ((1, 1), (20.6543, 100.0000)),
        ((0, 1), (0.0000, 79.3457)),
    ]
    for (hits, n), (expected_lo, expected_hi) in cases:
        lo, hi = wilson_interval(hits, n)
        assert lo == pytest.approx(expected_lo, abs=1e-3)
        assert hi == pytest.approx(expected_hi, abs=1e-3)


def test_wilson_interval_bounds_stay_within_0_and_100():
    lo, hi = wilson_interval(0, 50)
    assert lo >= 0.0
    lo, hi = wilson_interval(50, 50)
    assert hi <= 100.0


def test_wilson_interval_zero_n_returns_zero_zero():
    assert wilson_interval(0, 0) == (0.0, 0.0)
