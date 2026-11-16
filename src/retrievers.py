"""Retrieval methods and context engineering for Day 7 Agentic RAG."""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", ".")

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "starter") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "starter"))

from baseline import TfidfIndex, pack_context, tokenize  # noqa: E402


class SharedNeuralModels:
    """Singleton loader for cached local HuggingFace models to avoid repeated disk I/O."""
    _bi_tok = None
    _bi_mod = None
    _ce_tok = None
    _ce_mod = None

    @classmethod
    def warmup(cls) -> None:
        cls.encode(["warmup query"])
        cls.score_pairs([["warmup query", "warmup passage"]])

    @classmethod
    def encode(cls, texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> torch.Tensor:
        if cls._bi_tok is None or cls._bi_mod is None:
            cls._bi_tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            cls._bi_mod = AutoModel.from_pretrained(model_name, local_files_only=True).eval()
        with torch.no_grad():
            inp = cls._bi_tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            out = cls._bi_mod(**inp)
            mask = inp["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
            emb = (out[0] * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return F.normalize(emb, p=2, dim=1)

    @classmethod
    def score_pairs(cls, pairs: list[list[str]], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> list[float]:
        if not pairs:
            return []
        if cls._ce_tok is None or cls._ce_mod is None:
            cls._ce_tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            cls._ce_mod = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True).eval()
        with torch.no_grad():
            inp = cls._ce_tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
            logits = cls._ce_mod(**inp).logits.squeeze(-1).tolist()
        if isinstance(logits, float):
            return [logits]
        return [float(x) for x in logits]


def _match_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata") or {}
    for k, v in filters.items():
        if v is None:
            continue
        if str(meta.get(k, "")).strip().lower() != str(v).strip().lower():
            return False
    return True


class TfidfRetriever:
    """Method 1 (Baseline): Lexical TF-IDF without metadata filter."""
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = TfidfIndex(chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        return self.index.search(query, top_k=top_k)


class BM25Retriever:
    """Lexical Okapi BM25 retriever with optional metadata filtering."""
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75, use_filter: bool = True) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.use_filter = use_filter
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.tfs = [Counter(t) for t in self.doc_tokens]
        df = Counter(term for tf in self.tfs for term in tf)
        n_docs = len(chunks)
        self.idf = {term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("search query phải là chuỗi không rỗng")
        q_terms = tokenize(query)
        active_filters = filters if self.use_filter else None
        results: list[dict[str, Any]] = []
        for chunk, tf, dl in zip(self.chunks, self.tfs, self.doc_lens):
            if not _match_filters(chunk, active_filters):
                continue
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                freq = tf[term]
                idf = self.idf.get(term, 0.0)
                score += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl))
            if score > 0:
                results.append({**chunk, "score": round(score, 4), "strategy": "bm25"})
        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]


class DenseDay6Retriever:
    """Method 2: Dense Retrieval using Day 6 sentence-transformers/all-MiniLM-L6-v2 (384-dim)
    and Strategy C metadata pre-filtering. Supports live pgvector DB or persisted Day 6 vector index.
    """
    def __init__(self, chunks: list[dict[str, Any]], vector_index_path: Path | None = None, use_filter: bool = True) -> None:
        self.chunks = chunks
        self.use_filter = use_filter
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        if vector_index_path and vector_index_path.exists():
            payload = json.loads(vector_index_path.read_text(encoding="utf-8"))
            emb_map = {r["id"]: r["embedding"] for r in payload["records"]}
            ordered = [emb_map[c["id"]] for c in chunks]
            self.chunk_embs = torch.tensor(ordered, dtype=torch.float32)
        else:
            self.chunk_embs = SharedNeuralModels.encode([c["text"] for c in chunks])

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("search query phải là chuỗi không rỗng")
        q_emb = SharedNeuralModels.encode([query.strip()])[0]
        active_filters = filters if self.use_filter else None

        if self.database_url:
            try:
                import psycopg
                vec_str = "[" + ",".join(f"{float(x):.8f}" for x in q_emb.tolist()) + "]"
                sql = """
                    SELECT chunk_id, document_id, source, page, text, metadata,
                           1.0 - (embedding <=> %s::vector) AS score
                    FROM rag_chunks
                    WHERE strategy = 'C'
                """
                params: list[Any] = [vec_str]
                if active_filters:
                    for k, val in active_filters.items():
                        sql += " AND metadata->>%s = %s"
                        params.extend([k, str(val)])
                sql += " ORDER BY embedding <=> %s::vector ASC LIMIT %s;"
                params.extend([vec_str, top_k])
                with psycopg.connect(self.database_url, connect_timeout=2) as conn:
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        rows = cur.fetchall()
                        if rows:
                            return [
                                {
                                    "id": r[0],
                                    "source_id": r[1],
                                    "source": r[2],
                                    "page": r[3],
                                    "text": r[4],
                                    "metadata": r[5] if isinstance(r[5], dict) else json.loads(r[5]),
                                    "score": round(float(r[6]), 4),
                                    "strategy": "dense_day6_pgvector",
                                }
                                for r in rows
                            ]
            except Exception:
                pass

        scores = torch.mv(self.chunk_embs, q_emb).tolist()
        results: list[dict[str, Any]] = []
        for chunk, s in zip(self.chunks, scores):
            if not _match_filters(chunk, active_filters):
                continue
            results.append({**chunk, "score": round(float(s), 4), "strategy": "dense_day6"})
        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]


class DenseRerankRetriever:
    """Improvement 1: Dense Retrieval (top pool_k candidates) + Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)."""
    def __init__(self, dense_retriever: DenseDay6Retriever, pool_k: int = 10) -> None:
        self.dense = dense_retriever
        self.pool_k = pool_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        candidates = self.dense.search(query, filters=filters, top_k=max(self.pool_k, top_k))
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        logits = SharedNeuralModels.score_pairs(pairs)
        reranked = []
        for c, logit in zip(candidates, logits):
            reranked.append(
                {
                    **c,
                    "dense_score": c["score"],
                    "score": round(float(logit), 4),
                    "strategy": "dense_rerank",
                }
            )
        return sorted(reranked, key=lambda c: (-c["score"], c["id"]))[:top_k]


class HybridRRFRetriever:
    """Method 3 & Improvement 2: Hybrid Retrieval combining Dense Day 6 + BM25 via RRF,
    optionally followed by Cross-Encoder neural reranking and same-page neighbor retention.
    """
    def __init__(
        self,
        dense_retriever: DenseDay6Retriever,
        bm25_retriever: BM25Retriever,
        pool_k: int = 15,
        rrf_k: int = 60,
        use_reranker: bool = False,
    ) -> None:
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.pool_k = pool_k
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
        self.chunks_by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for c in dense_retriever.chunks:
            key = (c["source_id"], int(c["page"]))
            self.chunks_by_page.setdefault(key, []).append(c)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        d_hits = self.dense.search(query, filters=filters, top_k=max(self.pool_k, top_k))
        b_hits = self.bm25.search(query, filters=filters, top_k=max(self.pool_k, top_k))
        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for rank, c in enumerate(d_hits, start=1):
            cid = c["id"]
            by_id[cid] = c
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, c in enumerate(b_hits, start=1):
            cid = c["id"]
            by_id[cid] = c
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        merged = [
            {**by_id[cid], "rrf_score": round(sc, 6), "score": round(sc, 6), "strategy": "hybrid_rrf"}
            for cid, sc in rrf_scores.items()
        ]
        merged.sort(key=lambda c: (-c["score"], c["id"]))

        if not self.use_reranker or not merged:
            return merged[:top_k]

        top_cands = merged[: max(10, top_k)]
        logits = SharedNeuralModels.score_pairs([[query, c["text"]] for c in top_cands])
        reranked = []
        for c, logit in zip(top_cands, logits):
            prob = 1.0 / (1.0 + math.exp(-float(logit)))
            combined = 0.65 * prob + 0.35 * min(1.0, c["rrf_score"] * 30.0)
            reranked.append(
                {
                    **c,
                    "ce_logit": round(float(logit), 4),
                    "score": round(combined, 4),
                    "strategy": "hybrid_rrf_rerank",
                }
            )
        reranked.sort(key=lambda c: (-c["score"], c["id"]))

        # Same-page neighbor retention (GUIDE.md Section 3: "giữ đoạn lân cận"):
        # If Rank 1 is highly confident and its page has an adjacent chunk not yet in top_k,
        # include that adjacent chunk before low-confidence off-page chunks.
        selected = list(reranked[:top_k])
        if selected and top_k >= 2 and selected[0].get("ce_logit", -99.0) >= 1.5:
            top1 = selected[0]
            page_chunks = self.chunks_by_page.get((top1["source_id"], int(top1["page"])), [])
            existing_ids = {c["id"] for c in selected}
            for pc in page_chunks:
                if pc["id"] not in existing_ids and _match_filters(pc, filters):
                    nb_logit = SharedNeuralModels.score_pairs([[query, pc["text"]]])[0]
                    # Replace the weakest slot if the weakest slot has a negative CE logit
                    if selected[-1].get("ce_logit", 0.0) < 0.0:
                        neighbor_item = {
                            **pc,
                            "rrf_score": round(rrf_scores.get(pc["id"], 0.008), 6),
                            "ce_logit": round(float(nb_logit), 4),
                            "score": round(max(float( selected[-1]["score"] ) + 0.01, 0.55), 4),
                            "strategy": "hybrid_rrf_rerank",
                        }
                        selected[-1] = neighbor_item
                        selected.sort(key=lambda c: (-c["score"], c["id"]))
                        break

        return selected[:top_k]


def pack_context_smart(
    candidates: list[dict[str, Any]],
    budget: int = 800,
    sort_by_score: bool = False,
) -> tuple[str, list[str]]:
    """Context packing with deduplication and optional relevance-score ordering across multi-turn searches."""
    if not sort_by_score:
        return pack_context(candidates, budget=budget)

    best_by_id: dict[str, dict[str, Any]] = {}
    order_index: dict[str, int] = {}
    for idx, c in enumerate(candidates):
        cid = c["id"]
        order_index.setdefault(cid, idx)
        if cid not in best_by_id or float(c.get("score", 0.0)) > float(best_by_id[cid].get("score", 0.0)):
            best_by_id[cid] = c

    ordered = sorted(
        best_by_id.values(),
        key=lambda c: (-float(c.get("score", 0.0)), order_index[c["id"]], c["id"]),
    )
    return pack_context(ordered, budget=budget)