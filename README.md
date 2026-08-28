# Grounded RAG Assistant — Production Enterprise Intelligence

A production-grade, document-grounded Retrieval-Augmented Generation (RAG) system with a **persistent SQLite database**, **multi-session chat management**, a **multi-tier cognitive memory engine**, and **hybrid BM25 + dense retrieval**.

**100% Free Architecture**: Powered by Pinecone Serverless, OpenRouter `:free` model routing, and local MiniLM embeddings.

---

## 🌟 Key Features

* **Persistent SQLite Database (`data/rag_app.db`)**: Full WAL-mode database storing chat sessions, complete message histories, citations, chunk texts, and cognitive memory. Zero data is lost on reload or restarts.
* **Multi-Session Chat Manager**: Create new chats, switch between past conversations, rename, and delete chat threads in the sidebar.
* **Multi-Tier Cognitive Memory Engine**:
  * **Working Memory**: Recent conversation buffer + periodic LLM compression.
  * **Episodic Memory**: Past interaction recall using time-decayed vector similarity ($\text{Score} = \text{sim} \times e^{-\lambda \Delta t}$).
  * **Semantic Memory**: Persistent user preferences and domain fact graphs.
  * **Procedural Memory**: Pre-compiled task execution workflows for financial comparisons, tabular audits, and executive summaries.
* **Pinecone Serverless + Local MiniLM**: Vector similarity search in the cloud with zero token cost for embeddings and text storage.
* **OpenRouter Free Model Routing**: Dynamic task-based LLM routing (`gemma-3-27b:free`, `llama-3.3-70b:free`, `mistral-7b:free`) with automatic fallbacks and Gemini as last resort.
* **Grounding & Numerical Hallucination Audit**: Token-level claim verification and exact number auditing on every response.
* **Live Token Quota Meter**: Real-time daily token usage tracking and conservative quota budgeting.

---

## ⚙️ Quick Start & Setup

### 1. Clone & Navigate to Project
```bash
git clone https://github.com/WaqassKhn/Vector_RAG.git
cd Vector_RAG
```

### 2. Create & Activate Virtual Environment

#### On Windows (PowerShell / Command Prompt):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On Linux / macOS (Bash / Zsh):
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables (`.env`)
Copy `.env.example` to `.env` and fill in your keys:

```env
# Pinecone Serverless (Free starter account at pinecone.io)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-ntpc
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# OpenRouter (Free tier at openrouter.ai)
OPENROUTER_API_KEY=your_openrouter_api_key

# Google Gemini (Optional fallback)
GEMINI_API_KEY=your_gemini_api_key
```

---

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment (One-Command)

To run the entire stack in Docker with persistent database volumes:

```bash
# Start container with volume persistence
docker compose up -d --build

# View container logs
docker compose logs -f

# Stop containers (all database records & files remain safe in ./data)
docker compose down
```

---

## 🧪 Running Automated Tests

Run the full automated test suite:
```bash
pytest -v
```

Test suite coverage:
* `tests/test_database.py`: SQLite session lifecycle, message persistence, document/chunk storage, and token logs.
* `tests/test_cognitive_memory.py`: Episodic time decay, semantic preferences/facts, procedural recipes, and cognitive hub.
* `tests/test_openrouter_llm.py`: Model discovery, generation, and SSE streaming.
* `tests/test_pinecone_db.py`: Pinecone upsert, metadata filtering, and deletion.
* `tests/test_pipeline.py`: Parser, chunker, cleaner, hybrid reranker, and numerical grounding auditor.

---

## 📊 RAGBench Benchmark Evaluation

Evaluate retrieval and faithfulness against the official **RAGBench** benchmark:

```bash
python evaluation/eval_ragbench.py --subset covidqa --max_samples 10 --output_dir "ragbench eval score"
```

---

## 📁 Repository Structure

```
Vector_RAG/
├── config.py                 # Central configurations, model routing, and token budget
├── requirements.txt           # Python package dependencies
├── app.py                     # 3-Tab Streamlit Web Application
├── Dockerfile                 # Container image specification
├── docker-compose.yml         # Container compose with persistent data volume
├── .env.example               # Environment variables template
├── database/
│   ├── __init__.py
│   └── db_manager.py          # SQLite persistence manager (sessions, messages, chunks, memory)
├── pipeline/
│   ├── cleaner.py             # Unicode, line break & table formatting cleaner
│   ├── parser.py              # Multi-format document parser (PDF, CSV, XLSX, DOCX, TXT)
│   └── chunker.py             # Structure- and header-aware chunker
├── vectorstore/
│   ├── embeddings.py          # Local MiniLM-L6-v2 embeddings (0 API token cost)
│   └── pinecone_db.py         # Pinecone Serverless client backed by SQLite DB
├── rag/
│   ├── chain.py               # End-to-End RAG chain with streaming & cognitive injection
│   ├── reranker.py            # Hybrid BM25 + Vector Reciprocal Rank Fusion (RRF)
│   ├── openrouter_llm.py      # Multi-model router for OpenRouter free-tier LLMs
│   ├── llm.py                 # Gemini Flash SDK fallback
│   ├── cache.py               # Semantic Answer Cache
│   ├── token_counter.py       # Live TokenTracker & quota accounting
│   ├── agents/
│   │   ├── query_planner.py   # Decomposes complex queries with procedural guidance
│   │   └── merge_agent.py     # Synthesizes multi-retrieval results
│   └── memory/
│       ├── cognitive_hub.py   # Unified 4-tier cognitive memory coordinator
│       ├── conversation_memory.py # Working memory with LLM compression
│       ├── episodic_memory.py # Time-decayed past session recall
│       ├── semantic_memory.py # User preferences & domain fact graph
│       └── procedural_memory.py # Domain task execution workflows
├── evaluation/
│   ├── eval_ragbench.py       # RAGBench dataset evaluation runner
│   └── grounding_eval.py      # LLM-as-judge claim verification & numerical auditor
└── tests/
    ├── test_database.py       # Database CRUD & persistence tests
    ├── test_cognitive_memory.py # Multi-tier cognitive memory tests
    ├── test_openrouter_llm.py # OpenRouter LLM integration tests
    ├── test_pinecone_db.py    # Pinecone DB integration tests
    └── test_pipeline.py       # Pipeline component unit tests
```

---

:free models can be unavailable at any time, please check openrouter website for the current free models you can use for this project.

## 🔒 License & Credits

Built with [Streamlit](https://streamlit.io), [Pinecone](https://pinecone.io), [OpenRouter](https://openrouter.ai), and [Sentence-Transformers](https://sbert.net).
