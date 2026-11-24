"""Automated verification suite for datasets, benchmarks, and backend syntax."""

from __future__ import annotations

import ast
import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

import evaluate_predictions as evaluator  # noqa: E402


class TestProjectIntegrity(unittest.TestCase):
    def test_mfaq_datasets_present_and_valid(self) -> None:
        mfaq_dir = ROOT / "data" / "mfaq"
        for lang in ["en", "de", "es", "fr", "it"]:
            path = mfaq_dir / f"mfaq_{lang}_test.csv"
            self.assertTrue(path.is_file(), f"Missing MFAQ dataset: {path}")
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                self.assertIn("question", reader.fieldnames or [])
                self.assertIn("answer", reader.fieldnames or [])
                rows = list(reader)
            self.assertEqual(len(rows), 200, f"Expected 200 rows in {path.name}")

    def test_evaluation_metrics_and_model_ranking(self) -> None:
        summary = evaluator.evaluate_all(ROOT / "experiments" / "results")
        models = summary["models"]
        xlm = models["XLM-RoBERTa"]["overall"]
        mbert = models["mBERT"]["overall"]
        distil = models["DistilBERT"]["overall"]

        self.assertEqual(xlm["n"], 13790)
        self.assertGreater(xlm["pearson"], mbert["pearson"])
        self.assertGreater(mbert["pearson"], distil["pearson"])
        self.assertGreater(xlm["spearman"], mbert["spearman"])
        self.assertGreater(mbert["spearman"], distil["spearman"])
        self.assertLess(xlm["mse"], mbert["mse"])
        self.assertLess(mbert["mse"], distil["mse"])

        self.assertAlmostEqual(xlm["pearson"], 0.8273, places=3)
        self.assertAlmostEqual(xlm["spearman"], 0.8169, places=3)
        self.assertAlmostEqual(xlm["mse"], 0.7521, places=3)

    def test_backend_engine_syntax(self) -> None:
        engine_path = ROOT / "backend" / "semantic_engine.py"
        source = engine_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(engine_path))
        self.assertIsNotNone(tree)


if __name__ == "__main__":
    unittest.main()
