import unittest
from pathlib import Path

from src.agent_runner import RealNeuralController, run_agent_configurable
from src.retrievers import BM25Retriever, TfidfRetriever, pack_context_smart


class Day7PipelineTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "c1",
                "source_id": "art-of-war-1910",
                "source": "starter/data/art_of_war.pdf",
                "page": 4,
                "text": "The art of war is governed by five constant factors: (1) The Moral Law; (2) Heaven; (3) Earth; (4) The Commander; (5) Method and discipline.",
                "metadata": {"section": "Laying Plans"},
            },
            {
                "id": "c2",
                "source_id": "art-of-war-1910",
                "source": "starter/data/art_of_war.pdf",
                "page": 8,
                "text": "The total of a thousand ounces of silver per day. Such is the cost of raising an army of 100,000 men.",
                "metadata": {"section": "Waging War"},
            },
        ]

    def test_bm25_and_metadata_filter(self):
        bm25 = BM25Retriever(self.chunks, use_filter=True)
        hits = bm25.search("five constant factors", filters={"section": "Laying Plans"}, top_k=2)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "c1")
        empty = bm25.search("five constant factors", filters={"section": "Waging War"}, top_k=2)
        self.assertEqual(len(empty), 0)

    def test_pack_context_smart_orders_by_score_and_deduplicates(self):
        cands = [
            {**self.chunks[1], "score": 0.2},
            {**self.chunks[0], "score": 0.9},
            {**self.chunks[1], "score": 0.95},
        ]
        ctx, ids = pack_context_smart(cands, budget=1000, sort_by_score=True)
        self.assertEqual(ids, ["c2", "c1"])
        self.assertIn("[c2 |", ctx)

    def test_real_controller_answers_and_abstains(self):
        ret = TfidfRetriever(self.chunks)
        ctrl = RealNeuralController(max_searches=2)
        ans = run_agent_configurable(
            "What are the five constant factors that govern the art of war?",
            ctrl,
            ret,
            max_searches=2,
            top_k=2,
            budget=1000,
            sort_context_by_score=True,
        )
        self.assertEqual(ans["status"], "answered")
        self.assertIn("c1", ans["citations"])

        abstain = run_agent_configurable(
            "What is the exact chemical formula of gunpowder for naval cannons?",
            ctrl,
            ret,
            max_searches=2,
            top_k=2,
            budget=1000,
            sort_context_by_score=True,
        )
        self.assertEqual(abstain["status"], "abstain")
        self.assertEqual(abstain["citations"], [])


if __name__ == "__main__":
    unittest.main()