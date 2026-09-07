# Kết quả thực nghiệm và phân tích

Thư mục `results` chứa toàn bộ kết quả thực nghiệm của nhóm khi tái lập phương pháp **TableRAG**, kết quả của phương pháp cơ sở **NaiveRAG**, cùng các file tổng hợp và biểu đồ được sử dụng trong báo cáo và bài trình bày.

Các thí nghiệm được thực hiện trên hai bộ dữ liệu:

- **HybridQA**: bộ dữ liệu tiếng Anh được sử dụng để tái lập và so sánh TableRAG với NaiveRAG.
- **VinumQA**: bộ dữ liệu tiếng Việt được sử dụng để đánh giá khả năng áp dụng phương pháp trên dữ liệu tài chính tiếng Việt.

Các mô hình ngôn ngữ được sử dụng gồm:

- Claude
- DeepSeek
- Grok
- Qwen

---

## Cấu trúc thư mục

```text
results/
│
├── HybridQA_TABLERAG_results/
│   ├── TableRAG_claude.jsonl
│   ├── tableRAG_deepseek.csv
│   ├── TABLERAG_GROK.jsonl
│   └── tableRAG_qwen.csv
│
├── HybridQA_NaiveRAG_results/
│   ├── naiveRAG_claude.jsonl
│   ├── naiveRA_deepseek.csv
│   ├── NaiveRAG_grok.jsonl
│   └── NaiveRAG_qwen.csv
│
├── VinnumQA/
│   ├── Claude/
│   ├── Deepseek/
│   ├── Grok/
│   └── Qwen/
│
├── Figure2_error_distribution/
│   ├── figure2.png
│   └── HybridQA_Exact_Match_8_models.xlsx
│
├── Figure3_Distribution_of_reasoning-error_causes/
│   └── Các file phân tích nguyên nhân lỗi suy luận
│
└── Figure4_Figure5_iterations/
    ├── TableRAG_HybridQA_Accuracy_by_Iteration.png
    ├── TableRAG_HybridQA_Execution_Dynamics.xlsx
    └── TableRAG_HybridQA_Iteration_Distribution.png
