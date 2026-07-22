# ⚡ Grounded RAG Assistant (ChatGPT-Style UI & RAGBench Evaluation)

A robust Retrieval-Augmented Generation (RAG) system specialized in processing noisy text and numerical corporate documents (PDF, CSV, XLSX, DOCX, TXT), featuring a modern ChatGPT-style interface and RAGBench evaluation framework.

---

## 📂 How to Store Your Documents (Without Web App Uploads)

If you do not want to manually upload your files through the web interface every time you run the system, you can store your custom documents permanently inside the repository folder structure:

### Target Upload Directory
* **Folder Path**: [data/uploads/](file:///w:/RAG_NTPC/data/uploads/)
* **Supported File Types**: `.pdf`, `.csv`, `.xlsx`, `.docx`, `.txt`, `.md`

### Step-by-Step Guide:
1. Copy or move your custom PDF / document files into `data/uploads/`.
2. Ensure you are in the project root directory (`w:/RAG_NTPC`).
3. Run `streamlit run app.py`.
4. On startup, the RAG engine **automatically detects, cleans, chunks, and indexes** all files in `data/uploads/` into the local vector database ([data/vector_db/](file:///w:/RAG_NTPC/data/vector_db/)).
5. *Included Dummy Dataset*: A sample dirty corporate report ([sample_data/dirty_company_report.txt](file:///w:/RAG_NTPC/sample_data/dirty_company_report.txt)) is provided for instant testing out of the box.

---

## 🎨 ChatGPT-Style UI & Floating Island Settings

Launch the web interface from the project root directory:
```bash
# Make sure your terminal working directory is the project root (w:/RAG_NTPC)
streamlit run app.py
```

### Key UI Features:
- **ChatGPT Conversational Interface**: Rendered responses include grounded text, inline citations `[Source: Filename, Page X]`, expandable source context chunks, and live grounding audits.
- **Bottom Chat Box**: Type questions naturally in the fixed input box at the bottom of the screen (`st.chat_input`).
- **Floating Island Settings**: Click **"⚙️ System Parameters & Document Indexing"** at the top of the app to adjust Gemini API keys, chunk sizes, Top-K candidates, and trigger manual re-indexing of `data/uploads/`.

---

## ⚙️ Quick Start

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Set Gemini API Key
Create a `.env` file in the project root or export your environment variable:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 3. Run Automated Unit Tests
```bash
python -m pytest tests/test_pipeline.py
```

---

## 📊 RAGBench Evaluation

System performance is evaluated using the official **RAGBench** benchmark dataset (`rungalileo/ragbench`).

### Running RAGBench Evaluation
```bash
python evaluation/eval_ragbench.py --subset covidqa --max_samples 10 --output_dir "ragbench eval score"
```

### Command Arguments:
- `--subset`: RAGBench dataset subset (`covidqa`, `finqa`, `hotpotqa`, `msmarco`, `cuad`, etc.).
- `--max_samples`: Number of samples to evaluate (e.g. `10`, `50`).
- `--output_dir`: Output folder name for saved scores (default: `"ragbench eval score"`).

### Evaluation Outputs
Results are automatically exported to:
* **Output Folder**: [ragbench eval score/](file:///w:/RAG_NTPC/ragbench%20eval%20score/)
  * `ragbench_<subset>_eval_results.csv`: Per-sample predictions, faithfulness scores, and numerical audits.
  * `ragbench_<subset>_summary.json`: Benchmark pass rates and average grounding scores.

---

## 📄 Project Structure
```
w:/RAG_NTPC/
├── config.py                 # Central configurations & default parameters
├── requirements.txt           # Dependencies
├── app.py                     # ChatGPT-style Streamlit web application
├── data/
│   ├── uploads/               # 👈 PLACE YOUR PERMANENT DOCUMENTS HERE (.pdf, .csv, .docx, .txt)
│   └── vector_db/             # Local FAISS vector database
├── pipeline/
│   ├── cleaner.py             # Unicode, line break & table cleaner
│   ├── parser.py              # Multi-format document loader (PDF, CSV, DOCX, TXT)
│   └── chunker.py             # Context & header-aware chunker
├── vectorstore/
│   ├── embeddings.py          # Local SentenceTransformers & Gemini Embeddings
│   └── vector_db.py           # FAISS Vector Index & metadata storage
├── rag/
│   ├── reranker.py            # Hybrid BM25 + Vector RRF Reranker
│   ├── llm.py                 # Gemini Flash SDK integration with 429 backoff
│   └── chain.py               # Full Grounded RAG chain
├── evaluation/
│   ├── eval_ragbench.py       # Official RAGBench dataset evaluator
│   └── grounding_eval.py      # Claim verification & numerical accuracy auditor
├── ragbench eval score/       # 👈 RAGBENCH EVALUATION SCORES SAVED HERE
├── sample_data/
│   └── dirty_company_report.txt # 1 Sample Dummy Dataset for testing
└── tests/
    └── test_pipeline.py       # Automated unit tests
```
Frontend under progress
