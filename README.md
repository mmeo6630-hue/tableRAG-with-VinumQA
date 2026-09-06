# TableRAG — Notebook recorded and live walkthrough

**Bài báo:** *Reproducing TableRAG on HybridQA and a Vietnamese Evaluation*
**Phương pháp gốc:** Xiaohan Yu, Pu Jian, Chong Chen. *TableRAG: A Retrieval Augmented
Generation Framework for Heterogeneous Document Reasoning.* EMNLP 2025; arXiv:2506.10380.

---

## 0. Hai phần khác nhau trong dự án

| | File 1 — `tablerag_demo_envi.ipynb` - demo thực nghiệm| File 2 `tablerag_results.ipynb`— pipeline thực nghiệm quy mô đầy đủ |
|---|---|---|
| Vai trò | Diễn giải kiến trúc + chạy live trên quy mô nhỏ, **cả tiếng Anh lẫn tiếng Việt** | Sinh ra toàn bộ số liệu Bảng 1–3 trong bài báo |
| Quy mô | 1 ví dụ tiếng Anh + 1 ví dụ tiếng Việt (recorded + live) + câu hỏi tự chọn | 305 câu HybridQA + 50 câu VinumQA-50, × 4 backbone |
| Database | SQLite | MySQL/MariaDB |
| Retrieval | Lexical fallback (mặc định) hoặc cross-encoder thật (tùy chọn) | BGE-M3 + bge-reranker-v2-m3 + FAISS, đầy đủ |
| Backbone | 1 (Claude Haiku 4.5) | 4 (Claude Haiku 4.5, Qwen Flash, Grok Build 0.1, DeepSeek-V4-Flash) |
| Prompt | Bộ song ngữ: `PROMPTS` (Anh) và `PROMPTS_VI` (Việt), cùng chạy qua một hàm `tablerag_live()` | Prompt tiếng Việt riêng cho tập VinumQA-50 |

---

## 1. Cách chạy `tablerag_demo_live.ipynb` (File 1)

### 1.1. Tổng quan cấu trúc notebook

| Phần | Nội dung | Ngôn ngữ | Cần API key? | Tốn phí? |
|---|---|---|---|---|
| 1–2 | Dữ liệu mẫu tiếng Anh + xây 2 kho lưu trữ offline (SQLite + kho chunk) | Anh | Không | Không |
| 3–4 | Diễn giải lại (**replay**) một lần chạy TableRAG đã ghi sẵn trên ví dụ tiếng Anh, từng bước, kèm prompt thật; tổng kết chi phí/độ trễ | Anh | Không | Không |
| 5 | Chạy **live thật** lại đúng câu hỏi tiếng Anh ở Phần 3, đối chiếu với recorded run | Anh | **Có** | Có (~$0.01–0.05/câu) |
| **6** | **Ví dụ thứ hai — trên kho tiếng Việt (ViNumQA)**, dùng bộ prompt tiếng Việt riêng: 6.1 diễn giải recorded, 6.2 chạy live thật, 6.3 phân tích ý nghĩa | **Việt** | **Có** | Có |
| Cuối | Tổng kết toàn bài (cả 2 ví dụ) | — | — | — |

Phần 1–4 và 6.1 không cần key, không tốn phí. Phần 5 và 6.2 tự động bị bỏ qua nếu không có `ANTHROPIC_API_KEY`.

### 1.2. Chạy phần không cần API key (Phần 1–4 và 6.1)

1. Mở file bằng **Google Colab** (File → Upload notebook) hoặc **Kaggle Notebook** (Create → Notebook → Import).
2. Chọn **Run All** (Colab: `Runtime → Run all`; Kaggle: `Run All`).
3. Notebook sẽ tự động:
   - **Ví dụ tiếng Anh (Phần 1–4):** đọc dữ liệu JSON nhúng sẵn — câu hỏi *"What city hosted the World Championship in the season that Shouta Yasooka ranked 5th twice?"*, gold answer **Chiba** — dựng bảng SQLite thật, diễn giải lại từng bước (decompose → retrieve → SQL → combine) kèm đúng prompt gốc.
   - **Ví dụ tiếng Việt (Phần 6.1, hàm `walk()`):** đọc dữ liệu JSON nhúng sẵn của một bảng định giá cổ phiếu thật lấy từ **ViNumQA** — câu hỏi *"Tính giá mục tiêu trung bình dựa trên kết quả kèm tỷ trọng của FCFF và FCFE"*, gold answer **27688.5** — dựng lại đúng bảng đó trong SQLite, diễn giải lại từng bước bằng tiếng Việt, thực thi SQL live và đối chiếu với kết quả đã ghi.  

### 1.3. Chạy phần cần API key (Phần 5 và 6.2 — live thật)

**Bước 1 — Tạo API key**
Truy cập [console.anthropic.com](https://console.anthropic.com), tạo API key mới.

**Bước 2 — Thêm key vào môi trường**
- **Google Colab:** bấm biểu tượng chìa khóa 🔑 ở thanh bên trái → **Add new secret** → đặt tên chính xác là `ANTHROPIC_API_KEY` → dán giá trị key → bật toggle "Notebook access".
- **Kaggle:** vào **Add-ons → Secrets** → thêm secret cùng tên `ANTHROPIC_API_KEY` → khi mở notebook, bật quyền truy cập secret cho notebook đó.
- **Chạy local (Jupyter thường):** đặt biến môi trường trước khi mở notebook:
  ```bash
  export ANTHROPIC_API_KEY="sk-ant-..."
  jupyter notebook
  ```

**Bước 3 — Chạy Phần 5 (live, tiếng Anh)**
- Cell đầu Phần 5 tự động dò key theo thứ tự: Colab secrets → Kaggle secrets → biến môi trường.
- Nếu tìm thấy key, in ra `"live mode available - about $0.01-$0.05 per question"` và tự `pip install anthropic` nếu thư viện chưa có.
- Nếu không tìm thấy, in `"no ANTHROPIC_API_KEY found - the live sections below will be skipped"` — notebook vẫn chạy hết, không crash.
- Phần 5 chạy lại **đúng câu hỏi mẫu tiếng Anh**, gọi model thật, rồi so sánh: lần chạy live có đi cùng nhánh suy luận (SQL có được gọi hay bị bỏ qua) với lần chạy đã ghi sẵn hay không.

**Bước 4 — Chạy Phần 6.2 (live, tiếng Việt)**
- Cùng dùng key đã cấu hình ở Bước 2.
- Gọi `tablerag_live(EX_VI["question"], EX_VI, prompts=PROMPTS_VI)` — đây là điểm khác biệt kỹ thuật quan trọng: `tablerag_live()` nhận tham số `prompts` để chọn đúng bộ prompt tiếng Việt (`PROMPTS_VI`) thay vì bộ mặc định tiếng Anh (`PROMPTS`), nên **toàn bộ vòng lặp — phân rã câu hỏi, sinh SQL, tổng hợp câu trả lời — đều diễn ra bằng tiếng Việt**, không phải dịch máy phần hiển thị.
- Notebook tự chấm điểm gần đúng (sai số ≤1%) vì đáp án là một con số (27688.5), so với chấm "chứa gold answer" như ở ví dụ tiếng Anh.
- In ra: câu trả lời, số vòng lặp, số lần gọi API, chi phí ước tính.

### 1.4. Tùy chọn nâng cao

- **Bật retrieval thật (cross-encoder), thay vì lexical fallback:**
  Sửa `LIVE_RETRIEVAL = True` (cell định nghĩa `retrieve_live`). Lần chạy đầu sẽ tự tải mô hình `BAAI/bge-reranker-v2-m3` (~2.3 GB) — nên bật GPU runtime (`Runtime → Change runtime type → GPU` trên Colab) để không quá chậm. Áp dụng cho cả ví dụ tiếng Anh lẫn tiếng Việt.
- **Đổi backbone Claude khác:**
  Sửa dòng `BACKBONE = "claude-haiku-4-5"` (cell cấu hình key) thành model Claude khác, ví dụ `"claude-sonnet-4-6"`.
- **Đổi sang backbone khác họ (Qwen, Grok, DeepSeek):**
  Cần viết lại hàm `llm()` để gọi đúng SDK/endpoint của nhà cung cấp đó — đổi một dòng `BACKBONE` là không đủ vì định dạng request/response khác Anthropic.

---

## 2. Thực nghiệm quy mô đầy đủ (305 câu HybridQA + 50 câu VinumQA-50) — chạy trên File 2

**Quan trọng:** notebook `tablerag_demo_envi.ipynb` (File 1) **không** chạy toàn bộ 305 câu HybridQA hay 50 câu VinumQA-50 — nó chỉ minh họa cơ chế trên 2 ví dụ đơn lẻ (1 tiếng Anh, 1 tiếng Việt). Toàn bộ số liệu ở Bảng 1–3 của bài báo được tạo ra bởi **một pipeline riêng, quy mô đầy đủ `tablerag_results.ipynb`(File 2)**, dùng MySQL/MariaDB, BGE-M3 + reranker thật, và 4 backbone khác nhau, chạy theo đúng thiết lập ở của bài báo gốc.


### 2.1. Bộ dữ liệu HybridQA (305 câu, tiếng Anh)

Thực nghiệm dùng file `my_dev.json` (tập phát triển của HybridQA), xử lý 305 câu hỏi; so sánh cặp giữa TableRAG và NaiveRAG được tính trên 299 mã định danh xuất hiện ở cả hai đầu ra.

**Chuẩn bị dữ liệu:**
- Tải file `dev_excel.zip` từ [Google Drive]([HybridQA dataset](https://drive.google.com/drive/folders/1Pea6kiUZv0UP8k7Ohv19KorBdBaUrouE)) (theo định dạng của TableRAG gốc).
- Giải nén toàn bộ dữ liệu vào thư mục gốc của dự án trước khi chạy.

### 2.2. Bộ dữ liệu VinumQA-50 (50 câu, tiếng Việt)

Dataset là tập con 50 câu hỏi tiếng Việt, kèm 50 bảng Excel và 50 tài liệu JSON liên quan, chủ yếu thuộc lĩnh vực tài chính (tra cứu, lọc, so sánh, tính toán trên bảng).

**Nguồn dữ liệu (Kaggle):**
- [ViNumQA — kaggle.com/datasets/dintrn/vinumqa/data](https://www.kaggle.com/datasets/dintrn/vinumqa/data)

### 2.3. Thiết lập thực nghiệm đã dùng

- Backbone: Claude Haiku 4.5, Qwen Flash, Grok Build 0.1, DeepSeek-V4-Flash — gọi qua API tương ứng.
- Embedding: `BAAI/bge-m3` (đa ngôn ngữ); reranker: `BAAI/bge-reranker-v2-m3`.
- Chunk: 1.000 ký tự, overlap 200 ký tự; độ dài embedding tối đa 256 token.
- Index: FAISS `IndexFlatIP`, chạy CPU; embedding và reranker chạy trên GPU Colab.
- Truy xuất: recall 30 chunk → giữ 5 sau rerank → đưa 3 chunk cao nhất vào bước suy luận.
- Cơ sở dữ liệu bảng: MySQL hoặc MariaDB; tối đa 5 vòng lặp suy luận.
- NaiveRAG dùng cùng cấu hình embedding/rerank/chunk/top-k để đảm bảo so sánh công bằng, nhưng gửi thẳng 3 chunk cao nhất cho model, không qua SQL hay phân rã lặp.

### 2.4. Kết quả chính (tóm tắt, số liệu đầy đủ xem Bảng 1-3 trong bài báo)

| Backbone | NaiveRAG (HybridQA) | TableRAG (HybridQA) | TableRAG (VinumQA-50) |
|---|---|---|---|
| Claude Haiku 4.5 | 52.13% | 71.80% | 72.00% |
| Qwen Flash | 24.92% | 34.75% | 46.00% |
| Grok Build 0.1 | 15.41% | 39.02% | 34.00% |
| DeepSeek-V4-Flash | 37.70% | 42.95% | 56.00% |

Chi tiết độ trễ trung bình, số vòng lặp trung bình, và 10 nhóm lỗi đại diện (schema sai, SQL sai cú pháp/logic, nhầm lẫn viết tắt tài chính tiếng Việt, thiếu ngữ cảnh truy xuất...) được trình bày đầy đủ ở Bảng 3, figure 4 và mục 4 "Results and Discussion" của bài báo.

---

## 3. Mã nguồn và tài liệu đính kèm

**Tài liệu**
* **Bài báo thực nghiệm của nhóm:** `.docx` — "Reproducing TableRAG on HybridQA and an Experimental Evaluation in Vietnamese".

**Mã nguồn**
* **Kho lưu trữ chính của nhóm (GitHub):** [Github_Team](https://github.com/mmeo6630-hue/tableRAG-with-VinumQA) — Nơi chứa toàn bộ mã nguồn, tài liệu và pipeline thực nghiệm.
* **Notebook minh họa (File 1):** `tablerag_demo_envi.ipynb` (đã bao gồm trong kho lưu trữ/đính kèm filezip).
* **Pipeline thực nghiệm đầy đủ (File 2):** `tablerag_results` (đã bao gồm trong kho lưu trữ/đính kèm filezip).
* **Mã nguồn (tham khảo):** (https://github.com/yxh-y/TableRAG) từ kho lưu trữ TableRAG của tác giả.

**Bộ dữ liệu**
* **HybridQA gốc:** [wenhuchen/HybridQA](https://github.com/wenhuchen/HybridQA)
* **ViNumQA (tiếng Việt):** [Kaggle - dintrn/vinumqa](https://www.kaggle.com/datasets/dintrn/vinumqa/data)
