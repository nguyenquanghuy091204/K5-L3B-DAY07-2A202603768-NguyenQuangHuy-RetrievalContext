# Mã nguồn Day 7 — Agentic RAG & Context Engineering

## 1. Phần cơ bản

| File | Chức năng |
|---|---|
| `src/retrievers.py` | Hiện thực 5 bộ truy xuất / Retrieval (`TfidfRetriever`, `BM25Retriever`, `DenseDay6Retriever` dùng vector 384d `all-MiniLM-L6-v2` + filter `section`, `DenseRerankRetriever`, `HybridRRFRetriever` kết hợp Cross-Encoder `ms-marco-MiniLM-L-6-v2` + Neighbor Retention) và hàm `pack_context_smart`. |
| `src/agent_runner.py` | Hiện thực `RealNeuralController` (multi-turn search, query reformulation, abstention verification, grounded answer synthesis) và `run_agent_configurable`. |
| `src/run_experiments.py` | Chạy toàn bộ thực nghiệm Retrieval (5 cấu hình `dev` + 2 cấu hình `test`) và Agent (`dev` + `test`), sinh đầy đủ `traces.json`, `reviews.json`, `summary.json`. |
| `starter/prepare_day7_data.py` | Trích xuất 158 chunks (Strategy C), 20 câu hỏi có đáp án, 24 agent cases và lưu vector index 384 chiều từ Day 6. |

## 2. Phần Thử thách nâng cao

| File | Chức năng |
|---|---|
| `src/advanced/benchmark_data.py` | Bộ sinh dữ liệu benchmark 48 câu hỏi (18 dev / 30 test) bao phủ đủ 6 nhóm (table, diagram, chart, multi-hop, text, unanswerable) kèm gold evidence/absence reason. |
| `src/advanced/indexer.py` | Multimodal Indexer trích xuất 1.324 chunks (văn bản, bảng biểu, hình ảnh, biểu đồ), tạo 291 file ảnh crop PNG kèm tọa độ bbox và sinh vector index 384 chiều. |
| `src/advanced/retriever.py` | Bộ truy xuất đa phương thức kết hợp BM25 + Dense Vector RRF và multimodal context packing trong ngân sách 2.500 ký tự. |
| `src/advanced/agent.py` | Agent điều phối suy luận đa bước trên nội dung trực quan và bảng biểu, kiểm soát ngưỡng tin cậy để từ chối các câu hỏi ngoài phạm vi. |
| `src/advanced/run_advanced.py` | Thực thi đánh giá trên toàn bộ 30 câu test nâng cao, xuất đầy đủ `test_traces.json`, `test_reviews.json` và `test_summary.json` (đạt 96.7% ≥ 90%). |

## 3. Cách chạy tái lập (từ thư mục gốc của repo)

```bash
# 1. Chạy toàn bộ thực nghiệm cơ bản
python src/run_experiments.py

# 2. Chạy toàn bộ pipeline nâng cao
python src/advanced/benchmark_data.py
python src/advanced/indexer.py
python src/advanced/run_advanced.py

# 3. Chạy toàn bộ 21 bài unit test kiểm thử tự động
python -m unittest discover -s tests -v
```