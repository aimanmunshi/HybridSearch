"""Tests for the evaluation harness's scoring logic."""
from eval.run_eval import precision_at_k


def test_precision_at_k_counts_hits_within_top_k():
    assert precision_at_k(["a", "b", "c", "d", "e"], {"a", "c"}, k=5) == 2 / 5


def test_precision_at_k_ignores_hits_beyond_k():
    assert precision_at_k(["a", "b", "c", "d", "e", "f"], {"f"}, k=5) == 0.0


def test_precision_at_k_perfect_score():
    assert precision_at_k(["a", "b"], {"a", "b", "c"}, k=2) == 1.0


def test_precision_at_k_handles_fewer_results_than_k():
    # 1 hit out of a k=5 denominator, even though only 2 results were returned.
    assert precision_at_k(["a", "x"], {"a"}, k=5) == 1 / 5


def test_precision_at_k_empty_results_is_zero():
    assert precision_at_k([], {"a"}, k=5) == 0.0
