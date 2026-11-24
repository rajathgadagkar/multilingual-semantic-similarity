#!/usr/bin/env python3
"""
Reproducible Multilingual STS Benchmark Evaluator.

Computes Pearson correlation, Spearman rank correlation, Cosine similarity,
MSE, MAE, and RMSE — both overall (13,790 pairs) and per language (10 languages,
1,379 pairs each) — directly from the saved test-set prediction CSVs.

Zero external dependencies required (uses only Python standard library).

Author: Rajath Gadagkar <rajathgadagkar@gmail.com>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

LANGUAGES: List[Tuple[str, str]] = [
    ("de", "German"),
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("it", "Italian"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("zh", "Chinese"),
]

PAIRS_PER_LANGUAGE = 1379

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL_FILES = {
    "XLM-RoBERTa": "xlm_roberta_evaluation_results.csv",
    "mBERT": "mbert_evaluation_results.csv",
    "DistilBERT": "distilbert_evaluation_results.csv",
}


def rank_data(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    n = len(indexed)
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx * dy else 0.0


def spearman(x: List[float], y: List[float]) -> float:
    return pearson(rank_data(x), rank_data(y))


def cosine(x: List[float], y: List[float]) -> float:
    num = sum(a * b for a, b in zip(x, y))
    dx = math.sqrt(sum(a * a for a in x))
    dy = math.sqrt(sum(b * b for b in y))
    return num / (dx * dy) if dx * dy else 0.0


def compute_metrics(gold: List[float], pred: List[float]) -> Dict[str, float]:
    n = len(gold)
    mse_val = sum((g - p) ** 2 for g, p in zip(gold, pred)) / n
    mae_val = sum(abs(g - p) for g, p in zip(gold, pred)) / n
    return {
        "n": n,
        "pearson": round(pearson(gold, pred), 4),
        "spearman": round(spearman(gold, pred), 4),
        "cosine": round(cosine(gold, pred), 4),
        "mse": round(mse_val, 4),
        "mae": round(mae_val, 4),
        "rmse": round(math.sqrt(mse_val), 4),
    }


def evaluate_all(results_dir: Path = DEFAULT_RESULTS_DIR) -> Dict[str, object]:
    summary: Dict[str, object] = {"models": {}}
    for model_name, filename in MODEL_FILES.items():
        csv_path = results_dir / filename
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

        gold = [float(r["gold_similarity_score"]) for r in rows]
        pred = [float(r["predicted_similarity_score"]) for r in rows]

        per_lang: Dict[str, Dict[str, float]] = {}
        for idx, (code, _) in enumerate(LANGUAGES):
            start = idx * PAIRS_PER_LANGUAGE
            end = start + PAIRS_PER_LANGUAGE
            per_lang[code] = compute_metrics(gold[start:end], pred[start:end])

        summary["models"][model_name] = {
            "overall": compute_metrics(gold, pred),
            "per_language": per_lang,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multilingual STS-B test predictions across models and languages."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing model evaluation CSV files.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save computed metrics as JSON.",
    )
    args = parser.parse_args()

    summary = evaluate_all(args.results_dir)

    print("| Model | Test Pairs | Pearson | Spearman | Cosine | MSE | MAE | RMSE |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model_name, data in summary["models"].items():
        m = data["overall"]
        print(
            f"| {model_name} | {m['n']:,} | {m['pearson']:.4f} | "
            f"{m['spearman']:.4f} | {m['cosine']:.4f} | {m['mse']:.4f} | "
            f"{m['mae']:.4f} | {m['rmse']:.4f} |"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved metrics JSON to {args.json_out}")


if __name__ == "__main__":
    main()
