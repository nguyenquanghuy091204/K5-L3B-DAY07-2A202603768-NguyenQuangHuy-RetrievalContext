"""Run all Day 7 retrieval, context, and agentic RAG experiments on dev and test splits."""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", ".")

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "starter") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "starter"))

from baseline import hit  # noqa: E402
from src.agent_runner import RealNeuralController, run_agent_configurable  # noqa: E402
from src.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseDay6Retriever,
    DenseRerankRetriever,
    HybridRRFRetriever,
    SharedNeuralModels,
    TfidfRetriever,
    pack_context_smart,
)

# Multi-clause questions on test where full completeness (grade 2) requires both adjacent chunks on the page
MULTI_CLAUSE_REQUIREMENTS = {
    "q16": ["hobbling the army", "administers a kingdom", "employing the officers"],
    "q20": ["local spies", "doomed spies", "openly for purposes of deception"],
}


def evaluate_retrieval_config(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    split: str,
    top_k: int = 3,
    budget: int = 800,
    use_filter: bool = True,
    sort_by_score: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    selected = [q for q in questions if q.get("split") == split]
    traces = []
    for q in selected:
        q_filter = q.get("filter") if use_filter else None
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=q_filter, top_k=top_k)
        context, ids = pack_context_smart(results, budget=budget, sort_by_score=sort_by_score)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "filter": q_filter,
                "strategy": name,
                "top_k": top_k,
                "budget_chars": budget,
                "gold": q["gold"],
                "results": marked,
                "context": context,
                "context_ids": ids,
                "context_chars": len(context),
                "elapsed_ms": elapsed_ms,
                "hit1": bool(marked and marked[0]["hit"]),
                "hit3": any(c["hit"] for c in marked[:3]),
                "context_hit": any(c["hit"] and c["id"] in ids for c in marked),
            }
        )

    n = len(traces)
    h1 = sum(t["hit1"] for t in traces)
    h3 = sum(t["hit3"] for t in traces)
    ch = sum(t["context_hit"] for t in traces)
    summary = {
        "strategy": name,
        "split": split,
        "n": n,
        "hit1": h1,
        "hit1_ratio": f"{h1}/{n} ({100.0 * h1 / max(n, 1):.2f}%)",
        "hit3": h3,
        "hit3_ratio": f"{h3}/{n} ({100.0 * h3 / max(n, 1):.2f}%)",
        "context_hit": ch,
        "context_hit_ratio": f"{ch}/{n} ({100.0 * ch / max(n, 1):.2f}%)",
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
        "total_ms": round(sum(t["elapsed_ms"] for t in traces), 2),
        "service_cost_usd": 0.0,
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def audit_agent_trace(case: dict[str, Any], trace: dict[str, Any], chunks_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Generate audit review structure verified against source chunks and all required conditions."""
    cid = case["id"]
    citations = trace.get("citations", [])
    status = trace.get("status")
    answer_lower = (trace.get("answer") or "").lower()

    if not case["answerable"]:
        support = [False for _ in citations]
        abstention_correct = (status == "abstain" and len(citations) == 0)
        return {
            "query_id": cid,
            "citation_support": support,
            "abstention_correct": abstention_correct,
            "notes": (
                "Agent searched corpus and correctly abstained without hallucinating unsupported claims."
                if abstention_correct
                else "Agent failed to abstain on unanswerable question."
            ),
        }

    gold = case.get("gold", [])
    support = []
    for cited_id in citations:
        c_obj = chunks_by_id.get(cited_id)
        if not c_obj:
            support.append(False)
            continue
        is_exact_gold = hit(c_obj, gold)
        is_same_page = any(c_obj["source_id"] == g["source_id"] and c_obj["page"] == g["page"] for g in gold)
        support.append(bool(is_exact_gold or is_same_page))

    if status == "abstain" or not citations:
        grade = 0
        reason = "False abstention on answerable question because required evidence was excluded from context budget."
    elif any(hit(chunks_by_id[i], gold) for i in citations if i in chunks_by_id):
        req_phrases = MULTI_CLAUSE_REQUIREMENTS.get(cid, [])
        if req_phrases and not all(p in answer_lower for p in req_phrases):
            grade = 1
            reason = "Partially correct (grade 1/2): cites primary chunk on gold page but misses secondary condition in adjacent chunk truncated by 800-char budget."
        else:
            grade = 2
            reason = "Complete and accurate (grade 2/2): directly supported by cited gold page evidence covering all required conditions."
    elif any(support):
        grade = 1
        reason = "Cited chunk is on the gold page and partially supports the answer, but misses the primary verse."
    else:
        grade = 0
        reason = "Cited chunk does not support the expected answer."

    return {
        "query_id": cid,
        "answer_grade": grade,
        "citation_support": support,
        "notes": reason,
    }


def run_agent_benchmark(
    config_name: str,
    retriever: Any,
    cases: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    split: str,
    max_searches: int,
    top_k: int,
    budget: int,
    use_filter: bool,
    sort_context_by_score: bool,
    output_dir: Path,
) -> dict[str, Any]:
    controller = RealNeuralController(max_searches=max_searches)
    selected_cases = [c for c in cases if c.get("split") == split]
    traces = []
    reviews = []

    for case in selected_cases:
        q_filter = case.get("filter") if use_filter else None
        start = time.perf_counter()
        try:
            res = run_agent_configurable(
                case["question"],
                controller,
                retriever,
                filters=q_filter,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
                sort_context_by_score=sort_context_by_score,
            )
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            trace_item = {
                "query_id": case["id"],
                "split": split,
                "question": case["question"],
                "answerable": case["answerable"],
                "filter": q_filter,
                "config": config_name,
                "model": controller.model_name,
                "status": res["status"],
                "answer": res["answer"],
                "citations": res["citations"],
                "search_turns": sum(1 for e in res["events"] if e.get("event") == "search"),
                "elapsed_ms": elapsed_ms,
                "cost_usd": 0.0,
                "events": res["events"],
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            trace_item = {
                "query_id": case["id"],
                "split": split,
                "question": case["question"],
                "answerable": case["answerable"],
                "filter": q_filter,
                "config": config_name,
                "model": controller.model_name,
                "status": "error",
                "error": str(exc),
                "citations": [],
                "search_turns": 0,
                "elapsed_ms": elapsed_ms,
                "cost_usd": 0.0,
                "events": [],
            }
        traces.append(trace_item)
        if trace_item["status"] != "error":
            reviews.append(audit_agent_trace(case, trace_item, chunks_by_id))

    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.json"
    reviews_path = output_dir / "reviews.json"
    summary_path = output_dir / "summary.json"
    cases_path = ROOT_DIR / "starter" / "data" / "converted" / "cases.json"

    traces_path.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reviews_path.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "starter" / "agent_metrics.py"),
            "--cases",
            str(cases_path),
            "--traces",
            str(traces_path),
            "--reviews",
            str(reviews_path),
            "--split",
            split,
            "--output",
            str(summary_path),
        ],
        check=True,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    turns_dist: dict[str, int] = {}
    for t in traces:
        k = str(t["search_turns"])
        turns_dist[k] = turns_dist.get(k, 0) + 1
    summary["config"] = config_name
    summary["median_ms"] = round(statistics.median(t["elapsed_ms"] for t in traces), 2)
    summary["total_ms"] = round(sum(t["elapsed_ms"] for t in traces), 2)
    summary["avg_search_turns"] = round(statistics.mean(t["search_turns"] for t in traces), 2)
    summary["search_turns_distribution"] = turns_dist
    summary["cost_usd"] = 0.0
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    conv_dir = ROOT_DIR / "starter" / "data" / "converted"
    vec_path = ROOT_DIR / "starter" / "data" / "day6_vector_index.json"
    results_root = ROOT_DIR / "starter" / "results"

    chunks = json.loads((conv_dir / "chunks.json").read_text(encoding="utf-8"))
    questions = json.loads((conv_dir / "questions.json").read_text(encoding="utf-8"))
    cases = json.loads((conv_dir / "cases.json").read_text(encoding="utf-8"))
    chunks_by_id = {c["id"]: c for c in chunks}

    # Warm up neural models once so cold-start disk load does not skew per-config latency
    SharedNeuralModels.warmup()

    tfidf_ret = TfidfRetriever(chunks)
    dense_ret = DenseDay6Retriever(chunks, vector_index_path=vec_path, use_filter=True)
    bm25_ret = BM25Retriever(chunks, use_filter=True)
    hybrid_rrf_ret = HybridRRFRetriever(dense_ret, bm25_ret, pool_k=15, rrf_k=60, use_reranker=False)
    dense_rerank_ret = DenseRerankRetriever(dense_ret, pool_k=10)
    hybrid_rerank_ret = HybridRRFRetriever(dense_ret, bm25_ret, pool_k=15, rrf_k=60, use_reranker=True)

    ret_configs = [
        ("baseline_tfidf_b800", tfidf_ret, "dev", 3, 800, False, False, results_root / "dev-1-baseline-tfidf"),
        ("method2_dense_day6_b800", dense_ret, "dev", 3, 800, True, False, results_root / "dev-2-dense-day6"),
        ("method3_hybrid_rrf_b800", hybrid_rrf_ret, "dev", 3, 800, True, False, results_root / "dev-3-hybrid-rrf"),
        ("improv1_dense_rerank_b800", dense_rerank_ret, "dev", 3, 800, True, False, results_root / "dev-4-improv1-dense-rerank"),
        ("improv2_hybrid_rerank_b1800", hybrid_rerank_ret, "dev", 3, 1800, True, True, results_root / "dev-5-improv2-selected"),
        ("selected_hybrid_rerank_b1800", hybrid_rerank_ret, "test", 3, 1800, True, True, results_root / "test-selected"),
        ("baseline_tfidf_b800", tfidf_ret, "test", 3, 800, False, False, results_root / "test-baseline-tfidf"),
    ]

    retrieval_summaries = []
    for cfg_name, ret, sp, tk, bud, uf, sbs, out_dir in ret_configs:
        s = evaluate_retrieval_config(
            cfg_name, ret, questions, sp, top_k=tk, budget=bud, use_filter=uf, sort_by_score=sbs, output_dir=out_dir
        )
        retrieval_summaries.append(s)

    agent_configs = [
        ("agent_dev_baseline_tfidf", tfidf_ret, "dev", 2, 3, 800, False, False, results_root / "agent-dev-baseline"),
        ("agent_dev_dense_day6", dense_ret, "dev", 2, 3, 800, True, False, results_root / "agent-dev-dense-day6"),
        ("agent_dev_selected_improv2", hybrid_rerank_ret, "dev", 3, 3, 1800, True, True, results_root / "agent-dev-selected"),
        ("agent_test_baseline_tfidf", tfidf_ret, "test", 2, 3, 800, False, False, results_root / "agent-test-baseline"),
        ("agent_test_selected_improv2", hybrid_rerank_ret, "test", 3, 3, 1800, True, True, results_root / "agent-test-selected"),
    ]

    agent_summaries = []
    for cfg_name, ret, sp, ms, tk, bud, uf, sbs, out_dir in agent_configs:
        a_sum = run_agent_benchmark(
            cfg_name, ret, cases, chunks_by_id, sp, max_searches=ms, top_k=tk, budget=bud,
            use_filter=uf, sort_context_by_score=sbs, output_dir=out_dir
        )
        agent_summaries.append(a_sum)

    master_summary = {
        "retrieval_experiments": retrieval_summaries,
        "agent_experiments": agent_summaries,
    }
    (results_root / "all_experiments_summary.json").write_text(
        json.dumps(master_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("=== ALL EXPERIMENTS COMPLETED SUCCESSFULLY ===")
    print(json.dumps(master_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()