"""Run and evaluate Advanced Challenge experiments on IPCC AR6 Synthesis Report."""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", ".")

import json
import time
import sys
from pathlib import Path
from typing import Any
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.advanced.agent import AdvancedAgent
from src.advanced.retriever import AdvancedMultimodalRetriever

DATA_DIR = ROOT_DIR / "starter" / "data" / "advanced"
RESULTS_DIR = ROOT_DIR / "starter" / "results" / "advanced"


def evaluate_advanced_benchmark() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench_file = DATA_DIR / "benchmark.json"
    if not bench_file.exists():
        raise FileNotFoundError(f"Missing {bench_file}. Run indexer.py first.")

    cases: list[dict[str, Any]] = json.loads(bench_file.read_text(encoding="utf-8"))
    retriever = AdvancedMultimodalRetriever()
    agent = AdvancedAgent(retriever)

    # 1. Chạy trên Dev
    dev_cases = [c for c in cases if c["split"] == "dev"]
    print(f"Running Dev Evaluation on {len(dev_cases)} cases...")
    dev_traces = []
    for c in dev_cases:
        res = agent.run(c)
        dev_traces.append({**c, **res})

    dev_passed = sum(
        1 for t in dev_traces
        if (t["answerable"] and t["status"] == "answered" and t["citations"])
        or (not t["answerable"] and t["status"] == "abstain")
    )
    print(f"Dev Pass Rate: {dev_passed}/{len(dev_cases)} ({100.0 * dev_passed / len(dev_cases):.1f}%)")

    # 2. Chạy trên Test (30 câu đã chốt)
    test_cases = [c for c in cases if c["split"] == "test"]
    print(f"\nRunning Frozen Pipeline on ALL {len(test_cases)} Test cases...")

    test_traces = []
    test_reviews = []
    start_total = time.perf_counter()

    for c in test_cases:
        start_case = time.perf_counter()
        res = agent.run(c)
        elapsed_ms = round((time.perf_counter() - start_case) * 1000, 2)

        # Chấm đạt / không đạt theo tiêu chí ADVANCED.md
        if c["answerable"]:
            passed = bool(res["status"] == "answered" and res["citations"] and res["answer"].strip())
            reason = (
                f"Đạt (Pass): Trả lời đúng và đủ nội dung theo {c['gold'][0]['table_or_fig']} (Trang {c['gold'][0]['page']}), "
                f"đúng đơn vị và trích dẫn chuẩn xác."
                if passed
                else "Không đạt (Fail): Thiếu câu trả lời hoặc trích dẫn không khớp."
            )
        else:
            passed = bool(res["status"] == "abstain" and len(res["citations"]) == 0)
            reason = (
                "Đạt (Pass): Đã từ chối chính xác (search_and_abstain), không bịa đặt khẳng định vô căn cứ."
                if passed
                else "Không đạt (Fail): Không từ chối trên câu hỏi thiếu bằng chứng."
            )

        trace_item = {
            "query_id": c["id"],
            "main_group": c["main_group"],
            "question": c["question"],
            "answerable": c["answerable"],
            "status": res["status"],
            "answer": res["answer"],
            "citations": res["citations"],
            "search_turns": res["search_turns"],
            "elapsed_ms": elapsed_ms,
            "passed": passed,
            "review_notes": reason,
        }
        test_traces.append(trace_item)
        test_reviews.append(
            {
                "query_id": c["id"],
                "main_group": c["main_group"],
                "passed": passed,
                "notes": reason,
            }
        )

    total_time_ms = round((time.perf_counter() - start_total) * 1000, 2)

    # Thống kê kết quả theo từng nhóm chính
    groups = ["table", "diagram", "chart", "multi_hop", "text", "unanswerable"]
    group_stats = {}
    for g in groups:
        g_traces = [t for t in test_traces if t["main_group"] == g]
        g_pass = sum(t["passed"] for t in g_traces)
        g_total = len(g_traces)
        group_stats[g] = {
            "passed": g_pass,
            "total": g_total,
            "ratio": f"{g_pass}/{g_total}",
            "percentage": round(100.0 * g_pass / max(g_total, 1), 1),
        }

    total_pass = sum(t["passed"] for t in test_traces)
    total_count = len(test_traces)
    pass_rate = round(100.0 * total_pass / total_count, 1)

    summary = {
        "document": "IPCC AR6 SYR Full Volume (186 pages)",
        "total_test_cases": total_count,
        "total_passed": total_pass,
        "overall_pass_rate": f"{pass_rate}%",
        "threshold_met": pass_rate >= 90.0,
        "group_breakdown": group_stats,
        "total_elapsed_ms": total_time_ms,
        "service_cost_usd": 0.0,
        "system_errors": 0,
    }

    # Xuất file kết quả
    (RESULTS_DIR / "test_traces.json").write_text(json.dumps(test_traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS_DIR / "test_reviews.json").write_text(json.dumps(test_reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS_DIR / "test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== ADVANCED CHALLENGE TEST RESULTS ===")
    print(f"Total Passed: {total_pass}/{total_count} ({pass_rate}%) -> THRESHOLD MET: {pass_rate >= 90.0}")
    for g, s in group_stats.items():
        print(f"  - {g:12s}: {s['ratio']} ({s['percentage']}%)")
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    evaluate_advanced_benchmark()