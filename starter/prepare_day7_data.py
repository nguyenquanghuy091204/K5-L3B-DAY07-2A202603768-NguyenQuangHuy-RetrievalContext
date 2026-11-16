"""Prepare Day 7 converted chunks, 20 answerable benchmark questions, 24 agent cases,
and the 384-dim sentence-transformers/all-MiniLM-L6-v2 vector index from Day 6 Strategy C.
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", ".")

import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
DAY6_DIR = ROOT_DIR / "day06_repo"
if not DAY6_DIR.exists():
    alt_dir = ROOT_DIR.parent / "K5-L3B-DAY06-2A202603768-DataIngestion"
    if alt_dir.exists():
        DAY6_DIR = alt_dir

sys.path.insert(0, str(DAY6_DIR / "starter"))
from src.chunking import by_paragraph  # noqa: E402
from src.ingestion import load_pdf  # noqa: E402

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384

# Exact verbatim quotes in the extracted PDF text of Strategy C chunks
EXACT_QUOTES = {
    "q01": "(1) The Moral Law; (2) Heaven; (3) Earth; (4) The Commander; (5) Method and discipline.",
    "q02": "The Commander stands for the virtues of wisdom, sincerely, benevolence, courage and strict- ness.",
    "q04": "One cartload of the enemy’s provisions is equivalent to twenty of one’s own",
    "q05": "supreme excellence consists in breaking the enemy’s resistance without fighting.",
    "q06": "Thus the highest form of generalship is to balk the enemy’s plans; the next best is to prevent the junction of the enemy’s forces; the next in order is to attack the enemy’s army in the field; and the worst policy of all is to besiege walled cities.",
    "q07": "If you know the enemy and know yourself, you need not fear the result of a hun- dred battles. If you know yourself but not the enemy, for every victory gained you will also suffer a defeat.",
    "q09": "Energy may be likened to the bending of a cross- bow; decision, to the releasing of a trigger.",
    "q12": "The first is to burn soldiers in their camp; the sec- ond is to burn stores; the third is to burn baggage trains; the fourth is to burn arsenals and magazines; the fifth is to hurl dropping fire amongst the enemy.",
    "q13": "Having collected an army and concentrated his forces, he must blend and harmonize the different ele- ments thereof before pitching his camp.",
    "q15": "(1) Dispersive ground; (2) facile ground; (3) contentious ground; (4) open ground; (5) ground of intersecting highways; (6) serious ground; (7) difficult ground; (8) hemmed-in ground; (9) desperate ground.",
    "q20": "(1) Local spies; (2) inward spies; (3) converted spies; (4) doomed spies; (5) surviving spies.",
}

UNANSWERABLE_CASES = [
    {
        "id": "q21_unans_dev",
        "split": "dev",
        "question": "According to Sun Tzu in the chapter 'The Attack by Fire', what is the exact ratio of sulfur, saltpeter, and charcoal required to mix gunpowder for siege cannons?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Checked all 62 content pages (158 chunks), especially Chapter XII 'The Attack by Fire' (pages 59-61). The text only describes five incendiary methods using dry wood, reeds, and dropping fire; gunpowder chemicals (sulfur, saltpeter, charcoal) and cannons do not exist anywhere in the treatise.",
        "filter": {"section": "The Attack by Fire"},
    },
    {
        "id": "q22_unans_dev",
        "split": "dev",
        "question": "In the chapter 'Waging War', how many bronze coins per month is each conscripted foot soldier paid as a personal salary?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Checked all chunks in Chapter II 'Waging War' (pages 8-10) and the full corpus. Page 8 states the aggregate daily cost of raising 100,000 men (1,000 ounces of silver per day) and page 10 mentions rewarding soldiers who capture 10+ chariots, but individual monthly salaries in bronze coins are never mentioned.",
        "filter": {"section": "Waging War"},
    },
    {
        "id": "q23_unans_test",
        "split": "test",
        "question": "According to the treatise, how many oarsmen and boarding marines are assigned to a standard three-decked naval warship during a sea blockade?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Searched the entire 62-page treatise for naval terms ('warship', 'oarsmen', 'marines', 'naval', 'sea', 'blockade', 'ship', 'boat'). Aside from crossing rivers in Chapter IX (page 36) and a metaphor of a boat in Chapter XI (page 54), there is zero mention of naval warships or crew complements.",
        "filter": None,
    },
    {
        "id": "q24_unans_test",
        "split": "test",
        "question": "In which specific historical year and battle did Sun Tzu personally command the Wu army to capture the Chu capital of Ying?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Checked all 62 valid content pages (pages 4-65). While Chapter XI (page 53) mentions the mutual animosity of the states of Wu and Yueh as an illustration, the treatise contains only military doctrine and never states historical dates or Sun Tzu's personal campaign capturing Ying.",
        "filter": None,
    },
]


def embed_texts(texts: list[str], model_name: str = MODEL_NAME) -> list[list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).eval()
    all_vecs: list[list[float]] = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inp = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            out = model(**inp)
            mask = inp["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
            summed = torch.sum(out[0] * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            normed = F.normalize(summed / counts, p=2, dim=1)
            all_vecs.extend( [[round(float(x), 8) for x in row] for row in normed.tolist()] )
    return all_vecs


def main() -> None:
    pdf_path = DAY6_DIR / "data" / "art_of_war.pdf"
    bench_path = DAY6_DIR / "benchmark" / "benchmark.json"
    if not pdf_path.exists() or not bench_path.exists():
        raise FileNotFoundError(f"Missing Day 6 files at {DAY6_DIR}")

    raw_dir = ROOT_DIR / "starter" / "data" / "raw_day6"
    conv_dir = ROOT_DIR / "starter" / "data" / "converted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    conv_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract Strategy C chunks (by_paragraph max_size=600 with section metadata)
    pages = load_pdf(pdf_path, document_id="art-of-war-1910", source="starter/data/art_of_war.pdf")
    content_pages = [p for p in pages if p.metadata.get("is_content_page")]
    day6_chunks = []
    for p in content_pages:
        for c in by_paragraph(p, max_size=600):
            day6_chunks.append(
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "source": c.source,
                    "page": c.page,
                    "text": c.text,
                    "metadata": c.metadata,
                }
            )

    # 2. Load Day 6 benchmark questions and align exact verbatim substring quotes
    raw_bench = json.loads(bench_path.read_text(encoding="utf-8"))
    day6_questions = []
    for q in raw_bench["questions"]:
        q_copy = dict(q)
        if q_copy["id"] in EXACT_QUOTES:
            q_copy["evidence_quote"] = EXACT_QUOTES[q_copy["id"]]
        if q_copy.get("filter") is None:
            q_copy.pop("filter", None)
        day6_questions.append(q_copy)

    raw_chunks_file = raw_dir / "day6_chunks.json"
    raw_questions_file = raw_dir / "day6_questions.json"
    raw_chunks_file.write_text(json.dumps(day6_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_questions_file.write_text(json.dumps(day6_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 3. Run official starter/convert_day6.py
    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "starter" / "convert_day6.py"),
            "--chunks",
            str(raw_chunks_file),
            "--questions",
            str(raw_questions_file),
            "--output",
            str(conv_dir),
        ],
        check=True,
    )

    # 4. Build 24 agent cases (20 answerable + 4 unanswerable: 17 dev + 7 test)
    converted_questions = json.loads((conv_dir / "questions.json").read_text(encoding="utf-8"))
    cases = []
    for q in converted_questions:
        case_item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": q["gold"],
        }
        if "filter" in q and q["filter"]:
            case_item["filter"] = q["filter"]
        cases.append(case_item)

    for u in UNANSWERABLE_CASES:
        u_item = {k: v for k, v in u.items() if v is not None}
        cases.append(u_item)

    cases_path = conv_dir / "cases.json"
    cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 5. Compute and persist 384-dim vectors from sentence-transformers/all-MiniLM-L6-v2
    converted_chunks = json.loads((conv_dir / "chunks.json").read_text(encoding="utf-8"))
    vectors = embed_texts([c["text"] for c in converted_chunks], model_name=MODEL_NAME)
    vector_index_path = ROOT_DIR / "starter" / "data" / "day6_vector_index.json"
    vector_payload = {
        "model_name": MODEL_NAME,
        "dimensions": DIMENSIONS,
        "strategy": "C_by_paragraph_max600_section_filter",
        "chunk_count": len(converted_chunks),
        "records": [
            {**chunk, "embedding": vec}
            for chunk, vec in zip(converted_chunks, vectors)
        ],
    }
    vector_index_path.write_text(json.dumps(vector_payload, ensure_ascii=False) + "\n", encoding="utf-8")

    dev_ans = sum(1 for c in cases if c["split"] == "dev" and c["answerable"])
    dev_unans = sum(1 for c in cases if c["split"] == "dev" and not c["answerable"])
    test_ans = sum(1 for c in cases if c["split"] == "test" and c["answerable"])
    test_unans = sum(1 for c in cases if c["split"] == "test" and not c["answerable"])
    print(
        f"Prepared {len(converted_chunks)} chunks, {len(converted_questions)} answerable questions, "
        f"and {len(cases)} total agent cases (dev={dev_ans}+{dev_unans}, test={test_ans}+{test_unans})."
    )
    print(f"Saved 384-dim vector index to {vector_index_path}")


if __name__ == "__main__":
    main()