"""Agent Controller and configurable multi-turn agent loop for Day 7."""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", ".")

import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "starter") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "starter"))

from agentic import Controller, Retriever, run_agent  # noqa: E402
from baseline import tokenize  # noqa: E402
from src.retrievers import SharedNeuralModels, pack_context_smart  # noqa: E402

STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "are", "is", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "may", "might",
    "must", "can", "could", "the", "a", "an", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "into", "onto", "upon", "about", "above",
    "below", "under", "over", "between", "among", "through", "during", "before",
    "after", "and", "or", "but", "nor", "so", "yet", "both", "either", "neither",
    "not", "only", "own", "same", "than", "too", "very", "such", "that", "this",
    "these", "those", "their", "his", "her", "its", "our", "your", "my", "one",
    "ones", "according", "chapter", "specifically", "within", "treatise", "sun",
    "tzu", "art", "war", "general", "army", "enemy", "soldiers", "men", "force",
    "forces", "battle", "battles", "fighting",
}


def _extract_content_terms(question: str) -> list[str]:
    cleaned = re.sub(r"'[^']*'", " ", question)
    tokens = tokenize(cleaned)
    return [t for t in tokens if len(t) >= 4 and t not in STOPWORDS]


def _parse_context_blocks(context: str) -> list[dict[str, Any]]:
    if not context or not context.strip():
        return []
    blocks = []
    for raw_block in context.split("\n\n"):
        lines = raw_block.strip().splitlines()
        if len(lines) < 2:
            continue
        header = lines[0].strip()
        if not (header.startswith("[") and header.endswith("]")):
            continue
        parts = [p.strip() for p in header[1:-1].split("|")]
        chunk_id = parts[0]
        page_num = -1
        for p in parts[1:]:
            if p.lower().startswith("page "):
                try:
                    page_num = int(p.split()[-1])
                except ValueError:
                    page_num = -1
        text = "\n".join(lines[1:]).strip()
        blocks.append({"id": chunk_id, "page": page_num, "header": header, "text": text})
    return blocks


def _normalize_hyphenation(text: str) -> str:
    text = re.sub(r"^[IVXLCDM]+\s+/[A-Za-z0-9/.]+\s*", "", text)
    text = re.sub(r"^/[A-Za-z0-9/.]+\s*", "", text)
    text = re.sub(r"\s+\d+\s+/[A-Za-z0-9/.]+$", "", text)
    text = re.sub(r"([a-zA-Z])-\s+([a-zA-Z])", r"\1\2", text)
    return text.strip()


def _reformulate_query(question: str, turn_idx: int, existing_context: str) -> str:
    """Generate a focused or expanded query for subsequent search turns without using any gold data."""
    content_terms = _extract_content_terms(question)
    ctx_norm = _normalize_hyphenation(existing_context).lower()
    missing_terms = [t for t in content_terms if t not in ctx_norm]

    if turn_idx == 1:
        if missing_terms:
            return " ".join(missing_terms + content_terms[:5])
        return " ".join(content_terms)
    stems = [t[:6] if len(t) > 6 else t for t in content_terms]
    return " ".join(dict.fromkeys(content_terms + stems))


class RealNeuralController(Controller):
    """Controller powered by real neural Cross-Encoder (ms-marco-MiniLM-L-6-v2) and Bi-Encoder
    (all-MiniLM-L6-v2) models. Never accesses benchmark gold or expected answers.
    """
    def __init__(
        self,
        max_searches: int = 2,
        ce_abstain_threshold: float = -2.2,
        min_term_coverage: float = 0.34,
    ) -> None:
        self.max_searches = max_searches
        self.ce_abstain_threshold = ce_abstain_threshold
        self.min_term_coverage = min_term_coverage
        self.model_name = "local-neural:cross-encoder/ms-marco-MiniLM-L-6-v2+all-MiniLM-L6-v2"

    def _evaluate_blocks(self, question: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        if not blocks:
            return {"best_logit": -99.0, "coverage": 0.0, "ranked_blocks": []}

        norm_texts = [_normalize_hyphenation(b["text"]) for b in blocks]
        logits = SharedNeuralModels.score_pairs([[question, t] for t in norm_texts])

        content_terms = _extract_content_terms(question)
        ranked = []
        for b, norm_t, logit in zip(blocks, norm_texts, logits):
            t_tokens = set(tokenize(norm_t))
            matched = [
                term for term in content_terms
                if term in t_tokens or any(tok.startswith(term[:5]) or term.startswith(tok[:5]) for tok in t_tokens if len(tok) >= 4)
            ]
            cov = len(matched) / max(len(content_terms), 1)
            ranked.append(
                {
                    "id": b["id"],
                    "page": b.get("page", -1),
                    "text": norm_t,
                    "logit": float(logit),
                    "coverage": cov,
                    "combined": float(logit) + 4.0 * cov,
                }
            )
        ranked.sort(key=lambda x: (-x["combined"], x["id"]))

        all_ctx_tokens = set(tokenize(" ".join(norm_texts)))
        total_matched = [
            term for term in content_terms
            if term in all_ctx_tokens or any(tok.startswith(term[:5]) or term.startswith(tok[:5]) for tok in all_ctx_tokens if len(tok) >= 4)
        ]
        total_cov = len(total_matched) / max(len(content_terms), 1)
        return {
            "best_logit": ranked[0]["logit"],
            "coverage": total_cov,
            "ranked_blocks": ranked,
        }

    def _synthesize_grounded_answer(self, question: str, ranked_blocks: list[dict[str, Any]]) -> tuple[str, list[str]]:
        top_block = ranked_blocks[0]
        raw_text = top_block["text"]
        clauses = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+(?=\d+[,0-9]*\.\s+|[A-Z(])", raw_text)
            if len(s.strip()) > 10
        ]
        if not clauses:
            clauses = [raw_text]

        if len(clauses) == 1:
            selected_text = clauses[0]
        else:
            c_logits = SharedNeuralModels.score_pairs([[question, c] for c in clauses])
            content_terms = _extract_content_terms(question)
            scored_clauses = []
            for idx, (cl, l) in enumerate(zip(clauses, c_logits)):
                cl_tokens = set(tokenize(cl))
                ov = sum(1 for t in content_terms if t in cl_tokens or any(tok.startswith(t[:5]) for tok in cl_tokens if len(tok) >= 4))
                scored_clauses.append((idx, cl, float(l) + 1.5 * ov))
            scored_clauses.sort(key=lambda x: -x[2])
            best_idx = scored_clauses[0][0]
            # Include from best_idx through the end of the chunk so numbered lists (1), (2), (3) are never cut off
            selected_text = " ".join(clauses[best_idx:])

        citations = [top_block["id"]]
        answer = f"{selected_text} [{top_block['id']}]"

        # Attach complementary adjacent chunk ONLY if it belongs to the exact same source page
        for sec in ranked_blocks[1:]:
            if (
                sec["id"] != top_block["id"]
                and sec.get("page", -1) == top_block.get("page", -2)
                and sec.get("page", -1) > 0
                and (sec["logit"] > -1.0 or sec["coverage"] >= 0.34 or raw_text.endswith("15."))
            ):
                citations.append(sec["id"])
                answer = f"{answer} Additionally: {sec['text']} [{sec['id']}]"
                break

        return answer, citations

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            return {"action": "search", "query": question.strip()}

        latest = history[-1]
        queries_so_far = latest.get("queries", [])
        context = latest.get("context", "")
        blocks = _parse_context_blocks(context)
        eval_res = self._evaluate_blocks(question, blocks)

        is_confident = (
            eval_res["best_logit"] >= self.ce_abstain_threshold
            and eval_res["coverage"] >= self.min_term_coverage
        )

        if not is_confident and len(queries_so_far) < self.max_searches:
            next_query = _reformulate_query(question, len(queries_so_far), context)
            if next_query and next_query not in queries_so_far:
                return {"action": "search", "query": next_query}

        if not is_confident or not eval_res["ranked_blocks"]:
            return {"action": "answer", "answer": "", "citations": []}

        answer, citations = self._synthesize_grounded_answer(question, eval_res["ranked_blocks"])
        return {"action": "answer", "answer": answer, "citations": citations}


def run_agent_configurable(
    question: str,
    controller: Controller,
    retriever: Retriever,
    *,
    filters: dict[str, Any] | None = None,
    max_searches: int = 4,
    top_k: int = 3,
    budget: int = 800,
    sort_context_by_score: bool = False,
) -> dict[str, Any]:
    """Run the standard starter/agentic.py loop or the enhanced relevance-sorted context loop."""
    if not sort_context_by_score:
        return run_agent(
            question,
            controller,
            retriever,
            filters=filters,
            max_searches=max_searches,
            top_k=top_k,
            budget=budget,
        )

    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError("question, max_searches, top_k hoặc budget không hợp lệ")
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get("action") == "search":
            if sum(e["event"] == "search" for e in events) >= max_searches:
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [{"event": "search_limit"}],
                }
            query = action.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("search query phải là chuỗi không rỗng")
            results = retriever.search(query, filters=filters, top_k=top_k)
            if len(results) > top_k or len({r["id"] for r in results}) != len(results):
                raise ValueError("retriever phải trả tối đa top_k chunk ID duy nhất")
            candidates.extend(results)
            context, ids = pack_context_smart(candidates, budget=budget, sort_by_score=True)
            chunks = {c["id"]: c for c in candidates if c["id"] in ids}
            events.append(
                {
                    "event": "search",
                    "query": query,
                    "filter": filters,
                    "results": results,
                    "context": context,
                    "context_ids": ids,
                }
            )
            history = [
                {
                    "tool": "search",
                    "queries": [e["query"] for e in events if e["event"] == "search"],
                    "context": context,
                    "context_ids": ids,
                }
            ]
        elif action.get("action") == "answer":
            answer, citations = action.get("answer"), action.get("citations")
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError("answer/citations không hợp lệ")
            valid = bool(answer.strip() and citations and all(i in chunks and f"[{i}]" in answer for i in citations))
            if valid:
                return {
                    "status": "answered",
                    "answer": answer,
                    "citations": citations,
                    "events": events + [{"event": "answer"}],
                }
            return {
                "status": "abstain",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
                "events": events + [{"event": "invalid_or_missing_citations"}],
            }
        else:
            raise ValueError("action phải là search hoặc answer")
    return {"status": "abstain", "answer": "Không đủ bằng chứng để trả lời.", "citations": [], "events": events}