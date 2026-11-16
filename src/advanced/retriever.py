"""Multimodal retriever for IPCC AR6 Synthesis Report.
Combines Dense Day 6 embeddings, BM25, and Cross-Encoder neural reranker with modality-aware routing.
"""
import json
import math
from collections import Counter
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import torch
from src.retrievers import SharedNeuralModels

INDEX_PATH = ROOT_DIR / "starter" / "data" / "advanced" / "vector_index.json"


def tokenize_words(text: str) -> list[str]:
    import re
    return [w.lower() for w in re.findall(r"\b[A-Za-z0-9_.-]+\b", text) if len(w) >= 2]


class AdvancedMultimodalRetriever:
    def __init__(self, index_path: Path = INDEX_PATH) -> None:
        if not index_path.exists():
            raise FileNotFoundError(f"Missing vector index at {index_path}. Run indexer.py first.")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.records = payload["records"]
        self.chunk_embs = torch.tensor([r["embedding"] for r in self.records], dtype=torch.float32)

        # Lexical BM25 index
        self.doc_tokens = [tokenize_words(r["text"]) for r in self.records]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.tfs = [Counter(t) for t in self.doc_tokens]
        df = Counter(term for tf in self.tfs for term in tf)
        n_docs = len(self.records)
        self.idf = {term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(
        self,
        query: str,
        top_k: int = 5,
        target_modality: str | None = None,
        use_reranker: bool = True,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        # 1. Dense search
        q_emb = SharedNeuralModels.encode([query.strip()])[0]
        dense_scores = torch.mv(self.chunk_embs, q_emb).tolist()

        # 2. BM25 search
        q_terms = tokenize_words(query)
        bm25_scores = []
        for tf, dl in zip(self.tfs, self.doc_lens):
            sc = 0.0
            for t in q_terms:
                if t in tf:
                    freq = tf[t]
                    sc += self.idf.get(t, 0.0) * (freq * 2.5) / (freq + 1.5 * (0.25 + 0.75 * dl / self.avgdl))
            bm25_scores.append(sc)

        # 3. RRF Fusion
        dense_ranked = sorted(range(len(self.records)), key=lambda i: -dense_scores[i])
        bm25_ranked = sorted(range(len(self.records)), key=lambda i: -bm25_scores[i])

        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_ranked[:60], 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(bm25_ranked[:60], 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)

        # Modality boost
        candidates = []
        for idx, sc in rrf_scores.items():
            item = dict(self.records[idx])
            boost = 1.2 if (target_modality and item.get("chunk_type") == target_modality) else 1.0
            item["rrf_score"] = round(sc * boost, 6)
            candidates.append(item)

        candidates.sort(key=lambda x: -x["rrf_score"])
        pool = candidates[: max(15, top_k)]

        if not use_reranker or not pool:
            return pool[:top_k]

        # 4. Cross-Encoder rerank
        pairs = [[query, c["text"][:1000]] for c in pool]
        logits = SharedNeuralModels.score_pairs(pairs)
        for c, l in zip(pool, logits):
            c["ce_logit"] = round(float(l), 4)

        pool.sort(key=lambda x: -x["ce_logit"])
        return pool[:top_k]