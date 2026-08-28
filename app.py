"""
app.py — RAG_NTPC Production Application with Persistent Multi-Session Memory
─────────────────────────────────────────────────────────────────────────────
Features:
  - Persistent SQLite Database (data/rag_app.db): Zero data loss on interface close or reload.
  - Multi-Session Chat Manager: Gemini/ChatGPT-style flat sidebar conversation list.
  - Multi-Tier Cognitive Memory Engine: Working, Episodic, Semantic, and Procedural memory.
  - 3-Tab Minimalist UI:
      Tab 1: Chat      — streaming RAG chat with grounding audit, citations, and session history
      Tab 2: Documents — upload, index, and manage documents
      Tab 3: Settings  — cognitive memory explorer, model status, database controls
  - Sidebar: Gemini-style chat list + cognitive memory summary + active models.
"""

import os
import sys

# Disable broken legacy TensorFlow module to ensure PyTorch is used for all transformers embeddings
sys.modules["tensorflow"] = None
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is importable
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv()

ASSETS_DIR = BASE_DIR / "assets"
USER_AVATAR = str(ASSETS_DIR / "user_icon.svg") if (ASSETS_DIR / "user_icon.svg").exists() else "user"
ASSISTANT_AVATAR = str(ASSETS_DIR / "assistant_icon.svg") if (ASSETS_DIR / "assistant_icon.svg").exists() else "assistant"

from config import (
    VECTOR_DB_DIR, UPLOADS_DIR, DB_PATH,
    INITIAL_TOP_K, RERANKED_TOP_K,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
)
from database.db_manager import DatabaseManager, get_db
from pipeline.parser import DocumentParser
from pipeline.chunker import DocumentChunker
from vectorstore.embeddings import EmbeddingManager
from vectorstore.pinecone_db import PineconeDB
from rag.openrouter_llm import OpenRouterLLM
from rag.llm import GeminiLLM
from rag.reranker import HybridReranker
from rag.chain import RAGChain
from rag.memory.cognitive_hub import CognitiveMemoryHub
from rag.cache import SemanticAnswerCache
from rag.agents.query_planner import QueryPlannerAgent
from rag.agents.merge_agent import MergeAgent
from evaluation.grounding_eval import DocumentGroundingEvaluator
from rag.tracer import ExecutionTracer
from rag.error_handler import ErrorDiagnosticManager

logging.basicConfig(level=logging.WARNING)

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Assistant — Document Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Soft, minimalist, dark aesthetic
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    letter-spacing: -0.01em;
}

/* Background */
.stApp {
    background-color: #0d1117;
    background-image: radial-gradient(circle at 50% 0%, #161b22 0%, #0d1117 75%);
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] hr {
    margin: 1rem 0;
    border-color: rgba(255, 255, 255, 0.08);
}

/* Sidebar conversation buttons styling */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    text-align: left;
    justify-content: flex-start;
    padding: 6px 10px;
    min-height: 36px;
    font-size: 0.85rem;
    border-radius: 6px;
}

/* 1. New Chat Button - Green with Inverted SVG Icon */
div.st-key-new_chat_btn > button,
[data-testid="stSidebar"] div.st-key-new_chat_btn > button {
    background-color: #238636 !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    gap: 8px !important;
}

div.st-key-new_chat_btn > button::before,
[data-testid="stSidebar"] div.st-key-new_chat_btn > button::before {
    content: '';
    display: inline-block;
    width: 17px;
    height: 17px;
    background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KIDxwYXRoIGQ9Ik0xMiAxMy41VjcuNU05IDEwLjVIMTVNNyAxOFYyMC4zMzU1QzcgMjAuODY4NCA3IDIxLjEzNDggNy4xMDkyMyAyMS4yNzE2QzcuMjA0MjIgMjEuMzkwNiA3LjM0ODI3IDIxLjQ1OTkgNy41MDA1NCAyMS40NTk3QzcuNjc1NjMgMjEuNDU5NSA3Ljg4MzY3IDIxLjI5MzEgOC4yOTk3NiAyMC45NjAyTDEwLjY4NTIgMTkuMDUxOEMxMS4xNzI1IDE4LjY2MiAxMS40MTYyIDE4LjQ2NzEgMTEuNjg3NSAxOC4zMjg1QzExLjkyODIgMTguMjA1NSAxMi4xODQ0IDE4LjExNTYgMTIuNDQ5MiAxOC4wNjEzQzEyLjc0NzcgMTggMTMuMDU5NyAxOCAxMy42ODM3IDE4SDE2LjJDMTcuODgwMiAxOCAxOC43MjAyIDE4IDE5LjM2MiAxNy42NzNDMTkuOTI2NSAxNy4zODU0IDIwLjM4NTQgMTYuOTI2NSAyMC42NzMgMTYuMzYyQzIxIDE1LjcyMDIgMjEgMTQuODgwMiAyMSAxMy4yVjcuOEMyMSA2LjExOTg0IDIxIDUuMjc5NzYgMjAuNjczIDQuNjM4MDNDMjAuMzg1NCA0LjA3MzU0IDE5LjkyNjUgMy42MTQ2IDE5LjM2MiAzLjMyNjk4QzE4LjcyMDIgMyAxNy44ODAyIDMgMTYuMiAzSDcuOEM2LjExOTg0IDMgNS4yNzk3NiAzIDQuNjM4MDMgMy4zMjY5OEM0LjA3MzU0IDMuNjE0NiAzLjYxNDYgNC4wNzM1NCAzLjMyNjk4IDQuNjM4MDNDMyA1LjI3OTc2IDMgNi4xMTk4NCAzIDcuOFYxNEMzIDE0LjkzIDMgMTUuMzk1IDMuMTAyMjIgMTUuNzc2NUMzLjM3OTYyIDE2LjgxMTcgNC4xODgyNyAxNy42MjA0IDUuMjIzNTQgMTcuODk3OEM1LjYwNTA0IDE4IDYuMDcwMDMgMTggNyAxOFoiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+Cg==');
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
}

div.st-key-new_chat_btn > button:hover,
[data-testid="stSidebar"] div.st-key-new_chat_btn > button:hover {
    background-color: #2ea043 !important;
}

/* 2. Active Working Chat - Blue Hue */
[data-testid="stSidebar"] div[class*="st-key-chat_item_active_"] > button {
    background-color: rgba(56, 139, 253, 0.18) !important;
    color: #58a6ff !important;
    border: 1px solid rgba(56, 139, 253, 0.45) !important;
    font-weight: 600 !important;
    text-align: left !important;
    justify-content: flex-start !important;
}

[data-testid="stSidebar"] div[class*="st-key-chat_item_active_"] > button:hover {
    background-color: rgba(56, 139, 253, 0.28) !important;
    color: #79c0ff !important;
}

/* 3. Inactive Chats - Grey Hue */
[data-testid="stSidebar"] div[class*="st-key-chat_item_inactive_"] > button {
    background-color: rgba(255, 255, 255, 0.035) !important;
    color: #8b949e !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    font-weight: 400 !important;
    text-align: left !important;
    justify-content: flex-start !important;
}

[data-testid="stSidebar"] div[class*="st-key-chat_item_inactive_"] > button:hover {
    background-color: rgba(255, 255, 255, 0.08) !important;
    color: #e6edf3 !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
}

/* 4. Delete Chat Icon Button (Inverted White Trash Bin SVG) */
div[class*="st-key-del_sess_"] > button,
div[class*="st-key-del_doc_"] > button {
    background-color: rgba(255, 255, 255, 0.04) !important;
    background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KIDxwYXRoIGQ9Ik0xNiA2VjUuMkMxNiA0LjA3OTkgMTYgMy41MTk4NCAxNS43ODIgMy4wOTIwMkMxNS41OTAzIDIuNzE1NjkgMTUuMjg0MyAyLjQwOTczIDE0LjkwOCAyLjIxNzk5QzE0LjQ4MDIgMiAxMy45MjAxIDIgMTIuOCAySDExLjJDMTAuMDc5OSAyIDkuNTE5ODQgMiA5LjA5MjAyIDIuMjE3OTlDOC43MTU2OSAyLjQwOTczIDguNDA5NzMgMi43MTU2OSA4LjIxNzk5IDMuMDkyMDJDOCAzLjUxOTg0IDggNC4wNzk5IDggNS4yVjZNMyA2SDIxTTE5IDZWMTcuMkMxOSAxOC44ODAyIDE5IDE5LjcyMDIgMTguNjczIDIwLjM2MkMxOC4zODU0IDIwLjkyNjUgMTcuOTI2NSAyMS4zODU0IDE3LjM2MiAyMS42NzNDMTYuNzIwMiAyMiAxNS44ODAyIDIyIDE0LjIgMjJIOS44QzguMTE5ODQgMjIgNy4yNzk3NiAyMiA2LjYzODAzIDIxLjY3M0M2LjA3MzU0IDIxLjM4NTQgNS42MTQ2IDIwLjkyNjUgNS4zMjY5OCAyMC4zNjJDNSAxOS43MjAyIDUgMTguODgwMiA1IDE3LjJWNiIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K') !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 16px 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 6px !important;
    min-height: 36px !important;
    min-width: 34px !important;
    padding: 0 !important;
}

div[class*="st-key-del_sess_"] > button:hover,
div[class*="st-key-del_doc_"] > button:hover {
    background-color: rgba(248, 81, 73, 0.25) !important;
    border-color: rgba(248, 81, 73, 0.5) !important;
}

/* 5. Cancel / Erase Icon Button (Inverted White Circular X SVG) */
div[class*="st-key-del_pref_"] > button,
div[class*="st-key-del_fact_"] > button {
    background-color: rgba(255, 255, 255, 0.04) !important;
    background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KIDxwYXRoIGQ9Ik0xNSA5TDkgMTVNOSA5TDE1IDE1TTIyIDEyQzIyIDE3LjUyMjggMTcuNTIyOCAyMiAxMiAyMkM2LjQ3NzE1IDIyIDIgMTcuNTIyOCAyIDEyQzIgNi40NzcxNSA2LjQ3NzE1IDIgMTIgMkMxNy41MjI4IDIgMjIgNi40NzcxNSAyMiAxMloiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+Cg==') !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 16px 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 6px !important;
    min-height: 32px !important;
    min-width: 32px !important;
    padding: 0 !important;
}

div[class*="st-key-del_pref_"] > button:hover,
div[class*="st-key-del_fact_"] > button:hover {
    background-color: rgba(248, 81, 73, 0.25) !important;
    border-color: rgba(248, 81, 73, 0.5) !important;
}

div[class*="st-key-del_sess_"] button p,
div[class*="st-key-del_doc_"] button p,
div[class*="st-key-del_pref_"] button p,
div[class*="st-key-del_fact_"] button p {
    display: none !important;
}

/* Secondary general buttons outside sidebar */
button[kind="secondary"], .stButton > button[type="secondary"] {
    background-color: rgba(255, 255, 255, 0.05) !important;
    color: #c9d1d9 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px;
}

button[kind="secondary"]:hover, .stButton > button[type="secondary"]:hover {
    background-color: rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(22, 27, 34, 0.7);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: #8b949e;
    font-weight: 500;
    padding: 6px 18px;
    border: none;
    font-size: 0.9rem;
}

.stTabs [aria-selected="true"] {
    background: rgba(56, 139, 253, 0.15) !important;
    color: #58a6ff !important;
    border: 1px solid rgba(56, 139, 253, 0.3) !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: rgba(22, 27, 34, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    margin-bottom: 8px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(22, 27, 34, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 10px;
}

/* Badges */
.badge-pass {
    background: rgba(46, 160, 67, 0.2);
    color: #3fb950;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(46, 160, 67, 0.3);
}

.badge-fail {
    background: rgba(248, 81, 73, 0.2);
    color: #f85149;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(248, 81, 73, 0.3);
}

.badge-cache {
    background: rgba(56, 139, 253, 0.2);
    color: #58a6ff;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(56, 139, 253, 0.3);
}

/* Expanders */
details {
    background: rgba(22, 27, 34, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 4px 10px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    padding: 16px;
    background: rgba(22, 27, 34, 0.3);
}

/* Code blocks */
code {
    background: rgba(110, 118, 129, 0.15);
    color: #e6edf3;
    border-radius: 4px;
    padding: 2px 5px;
    font-size: 0.85em;
}

.model-live { color: #3fb950; font-size: 0.8rem; }
.model-down { color: #f85149; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ─── Cached singletons ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing database...")
def get_database_manager():
    return get_db()

@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_pinecone_db():
    db = get_database_manager()
    return PineconeDB(db=db)

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_manager():
    return EmbeddingManager(use_gemini=False)

@st.cache_resource(show_spinner="Connecting to OpenRouter...")
def get_openrouter_llm():
    return OpenRouterLLM()

@st.cache_resource
def get_reranker():
    return HybridReranker()


# ─── Session state init ───────────────────────────────────────────────────────

def init_session_state():
    db = get_database_manager()
    llm = get_openrouter_llm()

    if "cognitive_hub" not in st.session_state:
        st.session_state.cognitive_hub = CognitiveMemoryHub(llm=llm, db=db)
    if "cache" not in st.session_state:
        st.session_state.cache = SemanticAnswerCache()
    if "retrieval_top_k" not in st.session_state:
        st.session_state.retrieval_top_k = INITIAL_TOP_K
    if "rerank_top_k" not in st.session_state:
        st.session_state.rerank_top_k = RERANKED_TOP_K

    # Ensure active session exists in DB
    sessions = db.list_sessions(limit=1)
    if "active_session_id" not in st.session_state or not st.session_state.active_session_id:
        if sessions:
            st.session_state.active_session_id = sessions[0]["id"]
        else:
            st.session_state.active_session_id = db.create_session(title="New Conversation")


# ─── Indexing helper ──────────────────────────────────────────────────────────

def index_file(fpath: Path, vdb: PineconeDB, emb: EmbeddingManager) -> int:
    """Parses, chunks, embeds, and upserts one file. Returns chunk count."""
    parsed = DocumentParser.parse_file(str(fpath))
    chunker = DocumentChunker(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_parsed_document(parsed)
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    vecs = emb.embed_texts(texts)
    vdb.upsert_chunks(chunks, vecs)
    return len(chunks)


# ─── Sidebar (Gemini Style) ───────────────────────────────────────────────────

def render_sidebar():
    db = get_database_manager()
    with st.sidebar:
        st.markdown("### RAG Assistant")
        st.caption("Document Intelligence Engine")
        st.divider()

        # 1. New Chat Button (Green with SVG icon + text)
        if st.button("New Chat", key="new_chat_btn", use_container_width=True, type="primary"):
            new_sid = db.create_session(title="New Conversation")
            st.session_state.active_session_id = new_sid
            st.rerun()

        # 2. Gemini-Style Chat History List with direct inline delete
        st.markdown("##### Recent Chats")
        sessions = db.list_sessions(limit=30)
        current_sid = st.session_state.get("active_session_id")

        if sessions:
            for s in sessions:
                sid = s["id"]
                stitle = s["title"]
                is_active = (sid == current_sid)

                # Truncate title cleanly
                display_title = stitle if len(stitle) <= 22 else stitle[:20] + "..."

                col_chat, col_del = st.columns([0.84, 0.16])
                with col_chat:
                    btn_key = f"chat_item_active_{sid}" if is_active else f"chat_item_inactive_{sid}"
                    if st.button(
                        display_title,
                        key=btn_key,
                        use_container_width=True,
                        help=stitle,
                    ):
                        if sid != current_sid:
                            st.session_state.active_session_id = sid
                            st.rerun()

                with col_del:
                    if st.button("", key=f"del_sess_{sid}", help=f"Delete '{stitle}'", type="secondary"):
                        db.delete_session(sid)
                        remaining = db.list_sessions(limit=1)
                        st.session_state.active_session_id = remaining[0]["id"] if remaining else db.create_session(title="New Conversation")
                        st.rerun()

        st.divider()

        # 3. Cognitive Memory Summary Panel
        hub: CognitiveMemoryHub = st.session_state.cognitive_hub
        mem_summary = hub.get_dashboard_summary()
        with st.expander("Cognitive Memory Hub", expanded=False):
            st.markdown(f"**Working Memory:** `{mem_summary['working_memory']['recent_turns']}` turns")
            st.markdown(f"**Episodic Recall:** `{mem_summary['episodic_memory']['total_episodes']}` episodes")
            st.markdown(f"**User Preferences:** `{mem_summary['semantic_memory']['total_preferences']}` active")
            st.markdown(f"**Domain Facts:** `{mem_summary['semantic_memory']['total_facts']}` learned")
            st.markdown(f"**Task Recipes:** `{mem_summary['procedural_memory']['total_recipes']}` recipes")

        # 4. Active Models Status
        llm = get_openrouter_llm()
        with st.expander("Active Models", expanded=False):
            active = llm.get_active_models()
            for task, model in active.items():
                if model:
                    friendly_name = format_model_name(model)
                    st.markdown(
                        f'<span class="model-live">●</span> **{task}**: `{friendly_name}`',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span class="model-down">●</span> **{task}**: offline',
                        unsafe_allow_html=True,
                    )

        # 5. Quick Stats
        vdb = get_pinecone_db()
        stats = vdb.get_stats()
        cache: SemanticAnswerCache = st.session_state.cache
        cache_stats = cache.get_stats()

        col1, col2 = st.columns(2)
        col1.metric("Vectors", f"{stats['total_vector_count']:,}")
        col2.metric("Cache Hits", cache_stats["hit_count"])


# ─── Tab 1: Chat ──────────────────────────────────────────────────────────────

def render_chat_tab():
    db = get_database_manager()
    vdb = get_pinecone_db()
    emb = get_embedding_manager()
    llm = get_openrouter_llm()
    reranker = get_reranker()
    hub: CognitiveMemoryHub = st.session_state.cognitive_hub
    cache: SemanticAnswerCache = st.session_state.cache
    evaluator = DocumentGroundingEvaluator(llm=llm)

    rag_chain = RAGChain(
        vector_db=vdb,
        embedding_manager=emb,
        llm=llm,
        reranker=reranker,
    )

    known_docs = list(vdb.list_documents().keys())
    planner = QueryPlannerAgent(llm=llm, known_documents=known_docs)
    merger = MergeAgent(llm=llm)

    current_sid = st.session_state.active_session_id

    # Load persistent messages from SQLite DB
    persistent_messages = db.get_session_messages(current_sid)

    # Replay message history
    for msg in persistent_messages:
        role = msg["role"]
        avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            _render_message_extras(msg)

    # Chat input
    query = st.chat_input("Ask a question about your documents...")
    if not query:
        return

    # Auto-title session if it's currently default
    active_sess = db.get_session(current_sid)
    if active_sess and active_sess["title"] in ("New Conversation", "New Chat"):
        new_title = query[:28] + ("..." if len(query) > 28 else "")
        db.update_session_title(current_sid, new_title)

    # Save and display user message in DB
    db.save_message(
        session_id=current_sid,
        role="user",
        content=query,
    )
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        tracer = ExecutionTracer()
        try:
            # 1. Embed query
            embed_span = tracer.start_span("embed_query", component="embedding", inputs={"query": query})
            query_vec = emb.embed_query(query)
            tracer.finish_span(embed_span, outputs={"vector_dim": len(query_vec)})

            # 2. Check semantic cache
            cache_span = tracer.start_span("cache_lookup", component="cache")
            cache_hit = cache.lookup(query_vec)
            if cache_hit:
                tracer.finish_span(cache_span, outputs={"cache_hit": True, "similarity": cache_hit["cache_similarity"]})
                tracer.finish()
                st.markdown(cache_hit["answer"])
                st.markdown(
                    f'<span class="badge-cache">[Cache Hit: {cache_hit["cache_similarity"]:.1%}]</span>',
                    unsafe_allow_html=True,
                )

                if cache_hit.get("citations"):
                    st.markdown("**Citations:** " + ", ".join([f"`{c}`" for c in cache_hit["citations"]]))

                # Save assistant cache hit to database
                db.save_message(
                    session_id=current_sid,
                    role="assistant",
                    content=cache_hit["answer"],
                    citations=cache_hit["citations"],
                    grounding_score=1.0,
                    grounding_passed=True,
                    tokens_used=0,
                )
                hub.post_interaction_update(
                    query=query,
                    answer=cache_hit["answer"],
                    citations=cache_hit["citations"],
                    query_embedding=query_vec,
                    session_id=current_sid,
                )

                _render_execution_trace(tracer)
                return

            tracer.finish_span(cache_span, outputs={"cache_hit": False})

            # 3. Assemble Cognitive Memory Context
            mem_span = tracer.start_span("cognitive_memory_assembly", component="memory")
            cognitive_context = hub.build_cognitive_context(
                query=query,
                query_embedding=query_vec,
                session_id=current_sid,
            )
            tracer.finish_span(mem_span, outputs={"memory_context_len": len(cognitive_context)})

            # 4. Plan the query
            matched_recipe = hub.procedural_memory.match_recipe(query)
            procedural_hint = hub.procedural_memory.get_context_string(query) if matched_recipe else None

            planner_span = tracer.start_span("query_planning", component="query_planner", inputs={"query": query, "procedural_hint": bool(procedural_hint)})
            with st.spinner("Planning retrieval strategy..."):
                plan = planner.plan(query, procedural_hint=procedural_hint)
            tracer.finish_span(planner_span, outputs=plan)

            # 5. Execute retrieval + generation
            formatted_context = ""

            if plan["complexity"] == "simple" or len(plan["sub_queries"]) == 1:
                # Simple path: stream directly
                answer_parts = []
                answer_placeholder = st.empty()
                reranked_chunks = []
                citations = []

                for event_type, data in rag_chain.run_stream(
                    query=plan["sub_queries"][0],
                    initial_top_k=st.session_state.retrieval_top_k,
                    rerank_top_k=st.session_state.rerank_top_k,
                    filter_filenames=plan.get("doc_scope"),
                    memory_context=cognitive_context,
                    tracer=tracer,
                ):
                    if event_type == "context":
                        reranked_chunks = data
                    elif event_type == "token":
                        answer_parts.append(data)
                        answer_placeholder.markdown("".join(answer_parts) + "▌")
                    elif event_type == "done":
                        citations = data["citations"]
                        reranked_chunks = data.get("reranked_chunks", reranked_chunks)
                        formatted_context = data.get("formatted_context", "")

                answer = "".join(answer_parts)
                answer_placeholder.markdown(answer)

            else:
                # Complex path: parallel sub-queries → merge
                with st.spinner(f"Running {len(plan['sub_queries'])} retrieval passes..."):
                    sub_results = []
                    for sub_q in plan["sub_queries"]:
                        result = rag_chain.run(
                            query=sub_q,
                            initial_top_k=st.session_state.retrieval_top_k,
                            rerank_top_k=st.session_state.rerank_top_k,
                            filter_filenames=plan.get("doc_scope"),
                            memory_context=cognitive_context,
                            tracer=tracer,
                        )
                        sub_results.append(result)

                merge_span = tracer.start_span("merge_synthesis", component="merge_agent", inputs={"sub_queries_count": len(sub_results)})
                with st.spinner("Synthesizing answer..."):
                    merged = merger.merge(original_query=query, sub_results=sub_results)
                tracer.finish_span(merge_span, outputs={"citations_count": len(merged.get("citations", []))})

                answer = merged["answer"]
                reranked_chunks = merged["reranked_chunks"]
                citations = merged["citations"]
                formatted_context = merged.get("formatted_context", "")
                st.markdown(answer)

                if merged.get("sub_queries"):
                    with st.expander("Query Decomposition"):
                        for i, sq in enumerate(merged["sub_queries"], 1):
                            st.markdown(f"**Sub-query {i}:** {sq}")

            # Citations
            if citations:
                st.markdown("**Citations:** " + ", ".join([f"`{c}`" for c in citations]))

            # Context chunks expander
            with st.expander("Retrieved Context Chunks"):
                for idx, ch in enumerate(reranked_chunks):
                    st.markdown(f"**Chunk #{idx+1}** — `{ch['filename']}` Page {ch['page_number']}")
                    st.code(ch["text"], language="text")

            # Grounding audit
            eval_span = tracer.start_span("grounding_evaluation", component="eval")
            with st.spinner("Running grounding audit..."):
                eval_res = evaluator.evaluate_grounding(
                    query=query,
                    answer=answer,
                    retrieved_chunks=reranked_chunks,
                )
            tracer.finish_span(eval_span, outputs={"grounding_score": eval_res["overall_grounding_score"], "passed": eval_res["is_passed"]})

            badge = (
                '<span class="badge-pass">PASSED</span>'
                if eval_res["is_passed"]
                else '<span class="badge-fail">FAILED</span>'
            )
            with st.expander("Grounding Audit"):
                st.markdown(
                    f"**Score:** {eval_res['overall_grounding_score']*100:.1f}% | "
                    f"**Faithfulness:** {eval_res['faithfulness_score']*100:.1f}% | "
                    f"**Numerical:** {eval_res['numerical_accuracy_score']*100:.1f}% | "
                    f"Status: {badge}",
                    unsafe_allow_html=True,
                )
                if eval_res["unsupported_numbers"]:
                    st.error("Unsupported numbers: " + ", ".join(eval_res["unsupported_numbers"]))
                else:
                    st.success("Zero numerical hallucinations detected.")

            # Store in cache
            cache.store(query_vec, answer, citations, query_text=query)

            # Update Cognitive Memory Hub
            hub.post_interaction_update(
                query=query,
                answer=answer,
                citations=citations,
                query_embedding=query_vec,
                session_id=current_sid,
                model=getattr(llm, "last_model_used", "openrouter"),
                task="answer",
            )

            # Persist assistant message to SQLite Database
            db.save_message(
                session_id=current_sid,
                role="assistant",
                content=answer,
                citations=citations,
                grounding_score=eval_res["overall_grounding_score"],
                grounding_passed=eval_res["is_passed"],
                tokens_used=0,
            )

            # Complete and render execution trace
            tracer.finish()
            _render_execution_trace(tracer)

        except Exception as exc:
            tracer.finish()
            diag = ErrorDiagnosticManager.diagnose(exc, context="Query Processing Pipeline")
            st.markdown(diag.format_markdown(), unsafe_allow_html=True)
            _render_execution_trace(tracer)


def _render_execution_trace(tracer: ExecutionTracer):
    """Renders the step-by-step telemetry trace under assistant messages."""
    summary = tracer.get_summary()
    with st.expander("Execution Trace & Telemetry", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Latency", f"{summary['total_duration_ms']:,.0f} ms")
        c2.metric("LLM Calls", summary["llm_calls_count"])
        c3.metric("LLM Latency", f"{summary['total_llm_time_ms']:,.0f} ms")
        models_str = ", ".join([m.split("/")[-1].replace(":free", "") for m in summary["models_used"]]) or "local"
        c4.metric("Active Model", models_str)

        st.markdown("##### Execution Timeline")
        for idx, span in enumerate(tracer.spans, 1):
            status_text = "[OK]" if span.status == "success" else ("[WARN]" if span.status == "warning" else "[ERR]")
            col_s1, col_s2, col_s3 = st.columns([1, 4, 2])
            col_s1.markdown(f"**#{idx}** `{status_text}`")
            col_s2.markdown(f"`{span.component}`: **{span.name}**")
            col_s3.markdown(f"`{span.duration_ms:.1f} ms`")

            with st.expander(f"Details: {span.name}", expanded=False):
                if span.inputs:
                    st.markdown("**Inputs:**")
                    st.json(span.inputs)
                if span.outputs:
                    st.markdown("**Outputs:**")
                    st.json(span.outputs)
                if span.metadata:
                    st.markdown("**Metadata:**")
                    st.json(span.metadata)
                if span.error:
                    st.error(f"Error: {span.error}")


def _render_message_extras(msg: dict):
    """Renders citation/eval extras for replayed messages from database."""
    if msg.get("citations") and len(msg["citations"]) > 0:
        st.markdown("**Citations:** " + ", ".join([f"`{c}`" for c in msg["citations"]]))

    if msg.get("grounding_score", 0.0) > 0:
        passed = msg.get("grounding_passed", False)
        badge = '<span class="badge-pass">PASSED</span>' if passed else '<span class="badge-fail">FAILED</span>'
        with st.expander("Grounding Audit"):
            st.markdown(
                f"**Score:** {msg['grounding_score']*100:.1f}% | Status: {badge}",
                unsafe_allow_html=True,
            )


# ─── Tab 2: Documents ─────────────────────────────────────────────────────────

def render_documents_tab():
    vdb = get_pinecone_db()
    emb = get_embedding_manager()

    st.markdown("### Upload Documents")
    st.caption("Document ingestion and vector indexing use local embeddings (MiniLM-L6-v2) and cost zero API tokens.")
    uploaded = st.file_uploader(
        "Drag & drop files here",
        type=["pdf", "csv", "xlsx", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and st.button("Index Uploaded Files", type="primary"):
        progress = st.progress(0, text="Starting indexing...")
        total_chunks = 0
        has_error = False
        for i, f in enumerate(uploaded):
            save_path = UPLOADS_DIR / f.name
            try:
                with open(save_path, "wb") as w:
                    w.write(f.getbuffer())
                progress.progress((i + 0.5) / len(uploaded), text=f"Parsing {f.name}...")
                n = index_file(save_path, vdb, emb)
                total_chunks += n
                progress.progress((i + 1) / len(uploaded), text=f"Indexed {f.name} -> {n} chunks")
            except Exception as exc:
                has_error = True
                diag = ErrorDiagnosticManager.diagnose(exc, context=f"Uploading and Indexing Document `{f.name}`")
                st.markdown(diag.format_markdown(), unsafe_allow_html=True)

        progress.empty()
        if not has_error:
            st.success(f"Indexed {len(uploaded)} file(s) -> {total_chunks} total chunks saved to persistent database.")
            st.rerun()
        else:
            st.warning(f"Indexing completed with issues. Total indexed chunks: {total_chunks}")

    st.divider()

    # Document registry
    st.markdown("### Document Registry")
    registry = vdb.list_documents()

    if not registry:
        st.info("No documents indexed yet. Upload files above to get started.")
        return

    stats = vdb.get_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Vectors in Pinecone", f"{stats['total_vector_count']:,}")
    col2.metric("Documents in DB", stats.get("db_documents", len(registry)))
    col3.metric("Chunks in DB", f"{stats.get('db_chunks', 0):,}")

    st.markdown("")

    for filename, info in registry.items():
        col_name, col_chunks, col_date, col_del = st.columns([4, 1.5, 2.5, 1])
        col_name.markdown(f"`{filename}`")
        col_chunks.markdown(f"**{info['chunk_count']}** chunks")
        indexed_at = info.get("indexed_at", "—")[:10]
        col_date.caption(f"Indexed: {indexed_at}")
        if col_del.button("", key=f"del_doc_{filename}", help=f"Delete {filename}", type="secondary"):
            deleted = vdb.delete_by_filename(filename)
            st.success(f"Deleted {deleted} chunks for `{filename}`.")
            st.rerun()


# ─── Tab 3: Settings & Cognitive Memory Explorer ──────────────────────────────

MODEL_FRIENDLY_NAMES = {
    "openrouter/free": "OpenRouter Free Router (Dynamic Multi-Model)",
    "minimax/minimax-m3:free": "MiniMax M3 (Free)",
    "minimax/minimax-m2.7:free": "MiniMax M2.7 (Free)",
    "google/gemma-4-31b-it:free": "Google Gemma 4 31B (Free)",
    "google/gemma-4-26b-a4b-it:free": "Google Gemma 4 26B A4B (Free)",
    "inclusionai/ling-3.0-flash-fin:free": "InclusionAI Ling 3.0 Flash (Free)",
    "z-ai/glm-5.2:free": "Z.ai GLM 5.2 (Free)",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "NVIDIA Nemotron 3 Nano Omni (Free)",
    "nvidia/nemotron-3.5-lightning:free": "NVIDIA Nemotron 3.5 Lightning (Free)",
    "nvidia/nemotron-3-super-120b-a12b:free": "NVIDIA Nemotron 3 Super (Free)",
    "nvidia/nemotron-3-ultra-550b-a55b:free": "NVIDIA Nemotron 3 Ultra (Free)",
    "thinkingmachines/inkling:free": "Thinking Machines Inkling (Free)",
    "liquid/lfm-2.5-2.6b:free": "LiquidAI LFM 2.5 (Free)",
    "poolside/laguna-s-2.1:free": "Poolside Laguna S 2.1 (Free)",
    "cohere/north-mini-code:free": "Cohere North Mini Code (Free)",
}

def format_model_name(model_id: str) -> str:
    if not model_id:
        return "offline"
    if model_id in MODEL_FRIENDLY_NAMES:
        return MODEL_FRIENDLY_NAMES[model_id]
    cleaned = model_id.split("/")[-1].replace(":free", " (Free)")
    return cleaned.replace("-", " ").title()


def render_settings_tab():
    db = get_database_manager()
    llm = get_openrouter_llm()
    hub: CognitiveMemoryHub = st.session_state.cognitive_hub
    cache: SemanticAnswerCache = st.session_state.cache

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Cognitive Memory Explorer")
        
        # User Preferences Editor
        with st.expander("User Preferences (Semantic Memory)", expanded=True):
            prefs = hub.semantic_memory.get_all_preferences()
            for k, v in prefs.items():
                col_k, col_v, col_d = st.columns([2.5, 4, 1])
                col_k.markdown(f"**{k}**")
                col_v.caption(v)
                if col_d.button("", key=f"del_pref_{k}", help=f"Erase preference '{k}'", type="secondary"):
                    hub.semantic_memory.delete_preference(k)
                    st.rerun()

            st.markdown("##### Add / Update Preference")
            col_nk, col_nv = st.columns(2)
            new_k = col_nk.text_input("Key", placeholder="e.g. response_tone", key="new_pref_k")
            new_v = col_nv.text_input("Value", placeholder="e.g. Executive & concise", key="new_pref_v")
            if st.button("Save Preference") and new_k and new_v:
                hub.semantic_memory.set_preference(new_k, new_v)
                st.success(f"Saved preference `{new_k}`")
                st.rerun()

        # Domain Facts Graph
        with st.expander("Domain Facts Graph (Semantic Memory)", expanded=False):
            facts = hub.semantic_memory.get_all_facts(limit=20)
            if facts:
                for f in facts:
                    col_f, col_fd = st.columns([6, 1])
                    col_f.markdown(f"- **{f['subject']}** — *{f['predicate']}*: `{f['object']}`")
                    if col_fd.button("", key=f"del_fact_{f['id']}", help="Erase domain fact", type="secondary"):
                        hub.semantic_memory.delete_fact(f["id"])
                        st.rerun()
            else:
                st.caption("No domain facts stored yet. Learned facts will appear here.")

            st.markdown("##### Add Domain Fact")
            col_s, col_p, col_o = st.columns(3)
            subj = col_s.text_input("Subject", placeholder="NTPC", key="fact_s")
            pred = col_p.text_input("Predicate", placeholder="commercial_capacity", key="fact_p")
            obj = col_o.text_input("Object", placeholder="76 GW", key="fact_o")
            if st.button("Add Fact") and subj and pred and obj:
                hub.semantic_memory.add_fact(subj, pred, obj, source="manual_entry")
                st.success("Domain fact added.")
                st.rerun()

        # Procedural Task Recipes Catalog
        with st.expander("Task Execution Recipes (Procedural Memory)", expanded=False):
            recipes = hub.procedural_memory.get_all_recipes()
            for r in recipes:
                st.markdown(f"**Recipe:** `{r['name']}`")
                st.caption("Triggers: " + ", ".join([f"`{t}`" for t in r.get("trigger_patterns", [])]))
                for step in r.get("steps", []):
                    st.markdown(f"&nbsp;&nbsp;{step}")
                st.divider()

        # Episodic History
        with st.expander("Episodic Memory History", expanded=False):
            episodes = hub.episodic_memory.get_recent_episodes(limit=10)
            if episodes:
                for ep in episodes:
                    st.markdown(f"**Query:** *\"{ep['query']}\"*")
                    st.caption(f"Timestamp: {ep['timestamp'][:19]} | Citations: {len(ep.get('citations', []))}")
                    st.markdown(f"> {ep['answer'][:180]}...")
                    st.divider()
            else:
                st.caption("No past session episodes recorded yet.")

        st.divider()

        st.markdown("### Retrieval Parameters")
        st.session_state.retrieval_top_k = st.slider(
            "Initial retrieval top-K", 5, 30, st.session_state.retrieval_top_k
        )
        st.session_state.rerank_top_k = st.slider(
            "After reranking top-K", 2, 10, st.session_state.rerank_top_k
        )

    with col_right:
        st.markdown("### OpenRouter Model Status")
        active_models = llm.get_active_models()
        for task, model in active_models.items():
            if model:
                friendly_name = format_model_name(model)
                st.markdown(
                    f'<span class="model-live">●</span> **{task}** -> `{friendly_name}`',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="model-down">●</span> **{task}** -> offline',
                    unsafe_allow_html=True,
                )
        if not llm.is_available():
            st.warning("OpenRouter is unavailable. All requests will use Gemini as fallback.")
        else:
            st.success(f"OpenRouter connected — {len(llm._live_free_models)} free models available.")

        st.divider()

        st.markdown("### Database & Memory Controls")
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("Clear Answer Cache", use_container_width=True):
            cache.clear()
            st.success("Answer cache cleared.")
            st.rerun()

        if col_c2.button("Reset Cognitive Memory", use_container_width=True):
            hub.clear_all()
            st.success("Cognitive memory reset.")
            st.rerun()

        st.caption(f"SQLite Database Path: `{DB_PATH}`")

        st.divider()

        KEY_ICON_SVG = '<svg style="display:inline-block;vertical-align:middle;width:20px;height:20px;margin-left:6px;" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M15 9H15.01M15 15C18.3137 15 21 12.3137 21 9C21 5.68629 18.3137 3 15 3C11.6863 3 9 5.68629 9 9C9 9.27368 9.01832 9.54308 9.05381 9.80704C9.11218 10.2412 9.14136 10.4583 9.12172 10.5956C9.10125 10.7387 9.0752 10.8157 9.00469 10.9419C8.937 11.063 8.81771 11.1823 8.57913 11.4209L3.46863 16.5314C3.29568 16.7043 3.2092 16.7908 3.14736 16.8917C3.09253 16.9812 3.05213 17.0787 3.02763 17.1808C3 17.2959 3 17.4182 3 17.6627V19.4C3 19.9601 3 20.2401 3.10899 20.454C3.20487 20.6422 3.35785 20.7951 3.54601 20.891C3.75992 21 4.03995 21 4.6 21H6.33726C6.58185 21 6.70414 21 6.81923 20.9724C6.92127 20.9479 7.01881 20.9075 7.10828 20.8526C7.2092 20.7908 7.29568 20.7043 7.46863 20.5314L12.5791 15.4209C12.8177 15.1823 12.937 15.063 13.0581 14.9953C13.1843 14.9248 13.2613 14.8987 13.4044 14.8783C13.5417 14.8586 13.7588 14.8878 14.193 14.9462C14.4569 14.9817 14.7263 15 15 15Z" stroke="%2358a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        st.markdown(f"### API Keys (Masked) {KEY_ICON_SVG}", unsafe_allow_html=True)
        from config import PINECONE_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY
        def mask(k): return k[:8] + "..." + k[-4:] if len(k) > 12 else "not set"
        st.code(f"PINECONE_API_KEY   = {mask(PINECONE_API_KEY)}")
        st.code(f"OPENROUTER_API_KEY = {mask(OPENROUTER_API_KEY)}")
        st.code(f"GEMINI_API_KEY     = {mask(GEMINI_API_KEY)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    init_session_state()
    render_sidebar()

    st.markdown(
        "<h1 style='margin-bottom:4px;font-size:1.8rem;'>RAG Assistant</h1>"
        "<p style='color:#8b949e;margin-top:0;font-size:0.92rem;'>Persistent Document Intelligence — Powered by Pinecone + OpenRouter + SQLite</p>",
        unsafe_allow_html=True,
    )

    tab_chat, tab_docs, tab_settings = st.tabs(["Chat", "Documents", "Settings"])

    with tab_chat:
        render_chat_tab()

    with tab_docs:
        render_documents_tab()

    with tab_settings:
        render_settings_tab()


if __name__ == "__main__":
    main()
