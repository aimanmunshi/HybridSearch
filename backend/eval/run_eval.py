"""Evaluation harness: precision@5 for semantic-only vs hybrid search.

    python -m eval.run_eval

Ground truth (backend/eval/queries.json) was derived from objective corpus
metadata -- category/cuisine columns or an exact substring match -- rather
than by eyeballing what the search engine itself returns, which would make
the eval circular. See eval/README.md for the full methodology, including two
manual corrections made after a review pass caught community tags that were
simply wrong (a plain pasta dish and a jerk chicken recipe both tagged
'Curry').

Precision@k is used rather than recall or NDCG because it answers the
question a user actually has -- "were the first 5 things I saw worth
looking at?" -- without needing graded relevance judgments, which this
metadata-derived ground truth can't provide (a recipe either matches the
category/cuisine filter or it doesn't; there's no notion of "60% relevant").
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.db.connection import close_pool
from app.search.hybrid import hybrid_search
from app.search.rerank import rerank as rerank_results
from app.search.semantic import semantic_search

QUERIES_PATH = Path(__file__).parent / "queries.json"
RESULTS_DIR = Path(__file__).parent / "results"
K = 5
RERANK_CANDIDATE_K = 20

logging.basicConfig(level=logging.WARNING)  # keep eval output readable


def precision_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in expected_ids)
    return hits / k


def run() -> dict:
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]

    per_query = []
    for entry in queries:
        query = entry["query"]
        expected = set(entry["expected_ids"])

        semantic_results, _ = semantic_search(query, top_k=K)
        hybrid_results, _ = hybrid_search(query, top_k=K)
        rerank_candidates, _ = hybrid_search(query, top_k=RERANK_CANDIDATE_K)
        reranked_results = rerank_results(query, rerank_candidates, top_k=K)

        scores = {
            "semantic": precision_at_k([r.recipe_id for r in semantic_results], expected, K),
            "hybrid": precision_at_k([r.recipe_id for r in hybrid_results], expected, K),
            "hybrid+rerank": precision_at_k([r.recipe_id for r in reranked_results], expected, K),
        }
        per_query.append({"query": query, "expected_count": len(expected), **scores})
        print(f"{query:35s} semantic={scores['semantic']:.2f}  hybrid={scores['hybrid']:.2f}  hybrid+rerank={scores['hybrid+rerank']:.2f}")

    modes = ["semantic", "hybrid", "hybrid+rerank"]
    means = {mode: sum(q[mode] for q in per_query) / len(per_query) for mode in modes}

    return {"k": K, "num_queries": len(per_query), "mean_precision": means, "per_query": per_query}


def write_report(report: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Evaluation report (precision@{report['k']})",
        "",
        f"{report['num_queries']} queries. Mean precision@{report['k']}:",
        "",
        "| Mode | Mean precision@5 |",
        "|---|---|",
    ]
    for mode, score in report["mean_precision"].items():
        lines.append(f"| {mode} | {score:.3f} |")

    lines += ["", "## Per-query results", "", "| Query | Expected | Semantic | Hybrid | Hybrid+rerank |", "|---|---|---|---|---|"]
    for q in report["per_query"]:
        lines.append(
            f"| {q['query']} | {q['expected_count']} | {q['semantic']:.2f} | {q['hybrid']:.2f} | {q['hybrid+rerank']:.2f} |"
        )

    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        report = run()
        write_report(report)
        print("\nMean precision@5:")
        for mode, score in report["mean_precision"].items():
            print(f"  {mode:15s} {score:.3f}")
        print(f"\nReport written to {RESULTS_DIR / 'report.md'}")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
