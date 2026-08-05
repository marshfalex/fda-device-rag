import pytest

from fda_device_rag.eval.metrics import (
    hit_at_k,
    rank_of_first_gold_hit,
    reciprocal_rank,
    summarize_hit_rate,
    summarize_mrr,
    wilson_interval,
)


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


def test_rank_of_first_gold_hit_returns_one_indexed_rank():
    ranked = ["a", "b", "c", "d"]
    assert rank_of_first_gold_hit(ranked, ["c"]) == 3


def test_rank_of_first_gold_hit_returns_earliest_match_with_multi_chunk_gold():
    ranked = ["a", "b", "c", "d"]
    # "d" is gold too, but "b" (rank 2) is the best-ranked gold hit
    assert rank_of_first_gold_hit(ranked, ["d", "b"]) == 2


def test_rank_of_first_gold_hit_returns_none_when_absent():
    ranked = ["a", "b", "c"]
    assert rank_of_first_gold_hit(ranked, ["z"]) is None


def test_hit_at_k_true_within_k():
    assert hit_at_k(5, k=5) is True
    assert hit_at_k(1, k=5) is True


def test_hit_at_k_false_beyond_k():
    assert hit_at_k(6, k=5) is False


def test_hit_at_k_false_when_rank_is_none():
    assert hit_at_k(None, k=5) is False


def test_reciprocal_rank_of_a_rank():
    assert reciprocal_rank(4) == pytest.approx(0.25)


def test_reciprocal_rank_of_none_is_zero():
    assert reciprocal_rank(None) == 0.0


def test_summarize_hit_rate_counts_hits_within_k():
    # ranks: hit, hit, miss (beyond k), miss (absent)
    ranks = [1, 5, 6, None]
    summary = summarize_hit_rate(ranks, k=5)
    assert summary["hits"] == 2
    assert summary["n"] == 4
    assert summary["pct"] == pytest.approx(50.0)
    lo, hi = summary["wilson_ci_95"]
    assert 0.0 <= lo <= summary["pct"] <= hi <= 100.0


def test_summarize_hit_rate_empty_list_is_zero_over_zero():
    summary = summarize_hit_rate([], k=5)
    assert summary["hits"] == 0
    assert summary["n"] == 0
    assert summary["pct"] == 0.0
    assert summary["wilson_ci_95"] == (0.0, 0.0)


def test_summarize_mrr_averages_reciprocal_ranks():
    # 1/1, 1/4, 0 (absent) -> mean = (1 + 0.25 + 0) / 3
    ranks = [1, 4, None]
    assert summarize_mrr(ranks) == pytest.approx((1.0 + 0.25 + 0.0) / 3)


def test_summarize_mrr_empty_list_is_zero():
    assert summarize_mrr([]) == 0.0
