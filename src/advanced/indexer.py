"""Multimodal indexer for IPCC AR6 Synthesis Report (186 pages).
Extracts text chunks, tables, figures/diagrams/charts with bounding boxes and image crops.
Generates 384-dim normalized vector embeddings.
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", ".")

import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
PDF_PATH = ROOT_DIR / "data" / "advanced" / "IPCC_AR6_SYR_FullVolume.pdf"
OUT_DIR = ROOT_DIR / "starter" / "data" / "advanced"
CROPS_DIR = OUT_DIR / "crops"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384

# Danh sách các trang bìa, trang lót, trang phân tách phần không chứa nội dung phân tích hữu ích
UNHANDLED_OR_EMPTY_PAGES = [
    2, 5, 6, 8, 12, 18, 52, 53, 56, 57, 106, 132, 133, 134, 150, 151, 158, 159, 179
]

FIGURE_PATTERNS = [
    (r"(Figure\s+SPM\.[0-9]+[a-z]?)", "SPM"),
    (r"(Table\s+SPM\.[0-9]+[a-z]?)", "SPM"),
    (r"(Figure\s+[0-9]+\.[0-9]+[a-z]?)", "Section"),
    (r"(Table\s+[0-9]+\.[0-9]+[a-z]?)", "Section"),
]


def extract_table_markdown(table_data: list[list[str | None]]) -> str:
    cleaned_rows = []
    for r in table_data:
        row_cells = [str(c or "").strip().replace("\n", " ") for c in r]
        if any(row_cells):
            cleaned_rows.append(row_cells)
    if not cleaned_rows:
        return ""
    # Ensure uniform columns
    max_cols = max(len(r) for r in cleaned_rows)
    norm_rows = [r + [""] * (max_cols - len(r)) for r in cleaned_rows]
    header = "| " + " | ".join(norm_rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * max_cols) + " |"
    data = ["| " + " | ".join(r) + " |" for r in norm_rows[1:]]
    return "\n".join([header, sep] + data)


def embed_texts(texts: list[str]) -> list[list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True).eval()
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
            all_vecs.extend([[round(float(x), 8) for x in row] for row in normed.tolist()])
    return all_vecs


def build_advanced_index() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Missing PDF at {PDF_PATH}")

    doc = fitz.open(str(PDF_PATH))
    total_pages = len(doc)
    print(f"Loaded IPCC AR6 PDF: {total_pages} pages.")

    chunks: list[dict[str, Any]] = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]

        if page_num in UNHANDLED_OR_EMPTY_PAGES:
            continue

        raw_text = page.get_text() or ""
        if len(raw_text.strip()) < 50:
            continue

        # Determine report section
        if page_num <= 16:
            section_name = "Front Matter"
        elif 17 <= page_num <= 50:
            section_name = "Summary for Policymakers"
        elif 51 <= page_num <= 148:
            section_name = "Longer Report"
        else:
            section_name = "Annexes & Glossary"

        # 1. Trích xuất Tables nếu có
        try:
            tabs = list(page.find_tables())
        except Exception:
            tabs = []

        for t_idx, t in enumerate(tabs):
            t_data = t.extract()
            t_md = extract_table_markdown(t_data)
            bbox = [round(x, 1) for x in t.bbox]
            crop_name = f"table_p{page_num:03d}_{t_idx:02d}.png"
            crop_path = CROPS_DIR / crop_name

            # Render crop image
            try:
                pix = page.get_pixmap(clip=fitz.Rect(t.bbox), dpi=150)
                pix.save(str(crop_path))
            except Exception:
                pass

            table_id = f"ipcc_p{page_num:03d}_tab_{t_idx:02d}"
            chunks.append(
                {
                    "id": table_id,
                    "chunk_type": "table",
                    "source_id": "ipcc-ar6-syr",
                    "page": page_num,
                    "section": section_name,
                    "bbox": bbox,
                    "crop_image": f"starter/data/advanced/crops/{crop_name}",
                    "text": f"Table on Page {page_num} [{section_name}]:\n{t_md}",
                    "metadata": {"type": "table", "rows": len(t_data), "cols": len(t_data[0]) if t_data else 0},
                }
            )

        # 2. Trích xuất Figures / Charts / Diagrams
        fig_matches = re.findall(r"(Figure\s+SPM\.[0-9]+[a-z]?|Figure\s+[0-9]+\.[0-9]+[a-z]?|Table\s+SPM\.[0-9]+)", raw_text)
        unique_figs = list(dict.fromkeys(fig_matches))
        for f_name in unique_figs:
            f_clean = re.sub(r"\s+", "_", f_name)
            crop_name = f"{f_clean}_p{page_num:03d}.png"
            crop_path = CROPS_DIR / crop_name

            # Lưu crop trang đại diện
            try:
                pix = page.get_pixmap(dpi=120)
                pix.save(str(crop_path))
            except Exception:
                pass

            fig_type = "chart" if any(w in raw_text.lower() for w in ["warming", "emissions", "gtco2", "cost", "percentile"]) else "diagram"
            fig_id = f"ipcc_p{page_num:03d}_{f_clean.lower()}"
            caption_match = re.search(rf"{f_name}:([^\n\n]+(?:\n[^\n\n]+)*)", raw_text)
            caption = caption_match.group(0)[:1000] if caption_match else f"{f_name} on page {page_num}"

            chunks.append(
                {
                    "id": fig_id,
                    "chunk_type": fig_type,
                    "source_id": "ipcc-ar6-syr",
                    "page": page_num,
                    "section": section_name,
                    "bbox": [50, 50, 550, 750],
                    "crop_image": f"starter/data/advanced/crops/{crop_name}",
                    "text": f"{f_name} (Page {page_num}, {section_name}): {caption}",
                    "metadata": {"type": fig_type, "figure_name": f_name},
                }
            )

        # 3. Trích xuất Text blocks chuẩn
        blocks = page.get_text("blocks")
        for b_idx, b in enumerate(blocks):
            b_text = b[4].strip()
            if len(b_text) < 100 or b_text.startswith("Page ") or b_text.isdigit():
                continue
            b_id = f"ipcc_p{page_num:03d}_txt_{b_idx:02d}"
            chunks.append(
                {
                    "id": b_id,
                    "chunk_type": "text",
                    "source_id": "ipcc-ar6-syr",
                    "page": page_num,
                    "section": section_name,
                    "bbox": [round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)],
                    "crop_image": None,
                    "text": f"Page {page_num} [{section_name}]: {b_text}",
                    "metadata": {"type": "text", "char_count": len(b_text)},
                }
            )

    print(f"Generated {len(chunks)} multimodal chunks across {total_pages} pages.")

    # 4. Lưu Chunks & Benchmark
    from src.advanced.benchmark_data import ADVANCED_BENCHMARK
    (OUT_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "benchmark.json").write_text(json.dumps(ADVANCED_BENCHMARK, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 5. Tạo Vector Index
    print(f"Computing embeddings for {len(chunks)} chunks using {MODEL_NAME}...")
    texts_to_embed = [c["text"] for c in chunks]
    vectors = embed_texts(texts_to_embed)

    vector_payload = {
        "model_name": MODEL_NAME,
        "dimensions": DIMENSIONS,
        "total_chunks": len(chunks),
        "unhandled_pages": UNHANDLED_OR_EMPTY_PAGES,
        "records": [
            {**c, "embedding": vec}
            for c, vec in zip(chunks, vectors)
        ],
    }
    (OUT_DIR / "vector_index.json").write_text(json.dumps(vector_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Successfully saved chunks, benchmark, and vector index to {OUT_DIR}")


if __name__ == "__main__":
    build_advanced_index()