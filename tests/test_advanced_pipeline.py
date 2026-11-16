"""Unit tests for the Advanced Challenge pipeline on IPCC AR6 Synthesis Report."""
import unittest
from pathlib import Path
from src.advanced.benchmark_data import ADVANCED_BENCHMARK


class AdvancedPipelineTests(unittest.TestCase):
    def test_benchmark_spec_counts(self):
        dev = [c for c in ADVANCED_BENCHMARK if c["split"] == "dev"]
        test = [c for c in ADVANCED_BENCHMARK if c["split"] == "test"]
        self.assertGreaterEqual(len(dev), 18)
        self.assertGreaterEqual(len(test), 30)

        # Kiểm tra đủ 6 nhóm chính của Test theo ADVANCED.md
        tables = [c for c in test if c["main_group"] == "table"]
        diagrams = [c for c in test if c["main_group"] == "diagram"]
        charts = [c for c in test if c["main_group"] == "chart"]
        multi_hop = [c for c in test if c["main_group"] == "multi_hop"]
        text_cases = [c for c in test if c["main_group"] == "text"]
        unanswerable = [c for c in test if c["main_group"] == "unanswerable"]

        self.assertGreaterEqual(len(tables), 6)
        self.assertGreaterEqual(len(diagrams), 5)
        self.assertGreaterEqual(len(charts), 5)
        self.assertGreaterEqual(len(multi_hop), 4)
        self.assertGreaterEqual(len(text_cases), 4)
        self.assertGreaterEqual(len(unanswerable), 6)

    def test_unanswerable_cases_have_absence_reasons(self):
        unans = [c for c in ADVANCED_BENCHMARK if not c["answerable"]]
        for u in unans:
            self.assertIn("absence_reason", u)
            self.assertGreater(len(u["absence_reason"]), 20)

    def test_answerable_cases_have_gold_evidence(self):
        ans = [c for c in ADVANCED_BENCHMARK if c["answerable"]]
        for a in ans:
            self.assertIn("gold", a)
            self.assertGreaterEqual(len(a["gold"]), 1)
            self.assertGreater(a["gold"][0]["page"], 0)


if __name__ == "__main__":
    unittest.main()