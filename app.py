"""
app.py — CogniRAG: Production Application with Persistent Multi-Session Cognitive Memory
────────────────────────────────────────────────────────────────────────────────────────
Features:
  - Dark Minimal Palette: Deep obsidian black (#080c0a) and dark emerald green (#10b981).
  - Terminal Green Branding: "CogniRAG" styled in high-tech terminal phosphor green.
  - Persistent SQLite Database (data/rag_app.db): Zero data loss on interface close or reload.
  - Multi-Session Chat Manager: Flat sidebar conversation list with active session indicator.
  - Sidebar Navigation: Documents and Settings buttons in 1 line at the bottom of the dashboard.
  - Multi-Tier Cognitive Memory Engine: Working, Episodic, Semantic, and Procedural memory.
  - Clean Minimalist Main Chat: Uncluttered conversation view with grounding audit & citations.
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

# ─── Page config (must be first Streamlit command) ───────────────────────────
st.set_page_config(
    page_title="CogniRAG",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# ─── Dark Minimal (Dark Green & Black) CSS Design System ─────────────────────
st.markdown("""
<style>
:root {
    --app-bg: #080c0a;
    --app-bg-soft: #0e1511;
    --surface: #101713;
    --surface-soft: #16221c;
    --surface-muted: #1d2c24;
    --sidebar-bg: #060907;
    --sidebar-surface: #0c130f;
    --sidebar-surface-hover: #131e18;
    --text: #f3f4f6;
    --text-muted: #9ca3af;
    --text-subtle: #6b7280;
    --line: #18261e;
    --line-strong: #23372b;
    --accent: #10b981;
    --accent-hover: #059669;
    --accent-dark: #047857;
    --accent-soft: rgba(16, 185, 129, 0.08);
    --accent-border: rgba(16, 185, 129, 0.24);
    --terminal-green: #22c55e;
    --blue: #38bdf8;
    --green: #10b981;
    --red: #f87171;
    --warning: #fbbf24;
    --shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}

* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Helvetica, Arial, sans-serif;
    letter-spacing: -0.01em;
}

/* Base App Background & Text */
.stApp {
    background-color: var(--app-bg) !important;
    color: var(--text) !important;
}

[data-testid="stAppViewContainer"] {
    background-color: var(--app-bg) !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

.main .block-container,
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1040px;
    padding-top: 1rem;
    padding-bottom: 5rem;
}

/* Custom Minimal Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--app-bg);
}
::-webkit-scrollbar-thumb {
    background: var(--line-strong);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent-dark);
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    color: var(--text) !important;
    font-weight: 600;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 1.5rem;
    line-height: 1.25;
}

h3 {
    font-size: 1.15rem;
    margin-bottom: 0.4rem;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    line-height: 1.6;
    color: var(--text);
}

/* ─── Sidebar Styling ─── */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--line) !important;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: var(--sidebar-bg) !important;
}

.sidebar-header {
    padding: 0.35rem 0 0.85rem;
    margin-bottom: 0.85rem;
    border-bottom: 1px solid var(--line);
}

/* Terminal Green CogniRAG Title */
.sidebar-title {
    color: var(--terminal-green) !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'SF Mono', Consolas, monospace !important;
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1.2;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.01em;
    text-shadow: 0 0 10px rgba(34, 197, 94, 0.45);
}

.sidebar-desc {
    color: var(--text-muted);
    font-size: 0.78rem;
    line-height: 1.35;
    margin: 0;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h5 {
    color: var(--text) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
}

[data-testid="stSidebar"] hr {
    margin: 0.85rem 0;
    border-color: var(--line) !important;
}

hr {
    border-color: var(--line) !important;
    margin: 1rem 0;
}

/* ─── Buttons ─── */
.stButton > button {
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.88rem;
    transition: all 140ms ease;
    border: 1px solid var(--line-strong);
    background-color: var(--surface);
    color: var(--text);
}

.stButton > button:focus-visible {
    box-shadow: 0 0 0 2px var(--accent-border);
    outline: none;
}

/* New chat button & Primary buttons: ALWAYS solid black text & icon */
button[kind="primary"],
button[kind="primary"] *,
button[kind="primary"] p,
button[kind="primary"] span,
button[kind="primary"] div,
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-primary"] *,
button[data-testid="baseButton-primary"] p,
button[data-testid="baseButton-primary"] span,
button[data-testid="baseButton-primary"] svg,
.stButton > button[type="primary"],
.stButton > button[type="primary"] *,
.stButton > button[type="primary"] p,
.stButton > button[type="primary"] span,
div.st-key-new_chat_btn button,
div.st-key-new_chat_btn button *,
div.st-key-new_chat_btn button p,
div.st-key-new_chat_btn button span,
div.st-key-new_chat_btn button svg,
[data-testid="stSidebar"] div.st-key-new_chat_btn button,
[data-testid="stSidebar"] div.st-key-new_chat_btn button *,
[data-testid="stSidebar"] div.st-key-new_chat_btn button p,
[data-testid="stSidebar"] div.st-key-new_chat_btn button span,
[data-testid="stSidebar"] div.st-key-new_chat_btn button svg {
    background-color: #10b981 !important;
    color: #000000 !important;
    fill: #000000 !important;
    stroke: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    border: 1px solid #10b981 !important;
    font-weight: 700 !important;
}

/* Hover state for New chat / Primary buttons */
button[kind="primary"]:hover,
button[kind="primary"]:hover *,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="baseButton-primary"]:hover *,
.stButton > button[type="primary"]:hover,
.stButton > button[type="primary"]:hover *,
div.st-key-new_chat_btn button:hover,
div.st-key-new_chat_btn button:hover *,
[data-testid="stSidebar"] div.st-key-new_chat_btn button:hover,
[data-testid="stSidebar"] div.st-key-new_chat_btn button:hover * {
    background-color: #059669 !important;
    border-color: #059669 !important;
    color: #000000 !important;
    fill: #000000 !important;
    stroke: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* New chat button layout */
div.st-key-new_chat_btn > button,
[data-testid="stSidebar"] div.st-key-new_chat_btn > button {
    width: 100%;
    justify-content: center !important;
    text-align: center !important;
    font-size: 0.9rem !important;
    padding: 8px 12px !important;
}

/* Chat list items */
[data-testid="stSidebar"] div[class*="st-key-chat_item_active_"] > button {
    width: 100%;
    background-color: var(--accent-soft) !important;
    color: #34d399 !important;
    border: 1px solid var(--accent-border) !important;
    font-weight: 600 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 7px 10px !important;
    min-height: 38px !important;
}

[data-testid="stSidebar"] div[class*="st-key-chat_item_active_"] > button:hover {
    background-color: rgba(16, 185, 129, 0.16) !important;
    color: #6ee7b7 !important;
}

[data-testid="stSidebar"] div[class*="st-key-chat_item_inactive_"] > button {
    width: 100%;
    background-color: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid transparent !important;
    font-weight: 400 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 7px 10px !important;
    min-height: 38px !important;
}

[data-testid="stSidebar"] div[class*="st-key-chat_item_inactive_"] > button:hover {
    background-color: var(--sidebar-surface-hover) !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}

/* Delete buttons */
div[class*="st-key-del_sess_"] > button {
    background-color: transparent !important;
    color: var(--text-subtle) !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    min-height: 38px !important;
    min-width: 34px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[class*="st-key-del_sess_"] > button:hover {
    background-color: rgba(248, 113, 113, 0.12) !important;
    color: var(--red) !important;
    border-color: rgba(248, 113, 113, 0.25) !important;
}

div[class*="st-key-del_doc_"] > button,
div[class*="st-key-del_pref_"] > button,
div[class*="st-key-del_fact_"] > button {
    background-color: var(--surface) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
    min-height: 32px !important;
    min-width: 32px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[class*="st-key-del_doc_"] > button:hover,
div[class*="st-key-del_pref_"] > button:hover,
div[class*="st-key-del_fact_"] > button:hover {
    background-color: rgba(248, 113, 113, 0.12) !important;
    color: var(--red) !important;
    border-color: rgba(248, 113, 113, 0.25) !important;
}

button[kind="secondary"], .stButton > button[type="secondary"] {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--line-strong) !important;
}

button[kind="secondary"]:hover, .stButton > button[type="secondary"]:hover {
    background-color: var(--surface-soft) !important;
    color: var(--text) !important;
    border-color: var(--accent) !important;
}

/* Sidebar Nav Buttons (1 line, 2 columns at bottom) */
div.st-key-nav_docs_btn > button,
div.st-key-nav_set_btn > button,
div.st-key-nav_chat_btn > button {
    width: 100%;
    min-height: 38px;
    font-size: 0.84rem;
    padding: 6px 8px;
}

/* ─── Tabs Styling ─── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-radius: 0;
    padding: 0;
    gap: 1.5rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.25rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 0;
    color: var(--text-muted);
    font-weight: 500;
    padding: 0.5rem 0.2rem 0.75rem;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.95rem;
    transition: color 140ms ease, border-color 140ms ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text);
}

.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    font-weight: 600 !important;
}

/* ─── Chat Empty State ─── */
.chat-empty-state {
    text-align: center;
    padding: 3.5rem 1.5rem 2.5rem;
    max-width: 580px;
    margin: 0 auto;
}

.chat-empty-icon {
    font-size: 2.2rem;
    margin-bottom: 0.75rem;
    display: inline-block;
    color: var(--accent);
}

.chat-empty-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.4rem;
}

.chat-empty-desc {
    font-size: 0.88rem;
    color: var(--text-muted);
    line-height: 1.5;
    margin-bottom: 1.75rem;
}

.prompt-chips {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    text-align: left;
}

.prompt-chip {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    font-size: 0.85rem;
    color: var(--text-muted);
    transition: border-color 140ms ease, background 140ms ease;
}

.prompt-chip:hover {
    border-color: var(--accent-border);
    background: var(--surface-soft);
    color: var(--text);
}

/* ─── Chat Messages ─── */
[data-testid="stChatMessage"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    box-shadow: var(--shadow);
    margin-bottom: 0.75rem !important;
    padding: 0.85rem 1rem !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: var(--text) !important;
}

[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border-top: 1px solid var(--line) !important;
    border-radius: 8px !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-border) !important;
}

/* ─── Metric Cards ─── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: var(--shadow);
    padding: 0.7rem 0.85rem;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.35rem !important;
    font-weight: 650 !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}

[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: var(--sidebar-surface);
    border-color: var(--line);
    box-shadow: none;
}

[data-testid="stSidebar"] [data-testid="stMetricValue"],
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: var(--text);
}

/* ─── Badges ─── */
.badge-pass {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(16, 185, 129, 0.28);
}

.badge-fail {
    background: rgba(248, 113, 113, 0.12);
    color: #f87171;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(248, 113, 113, 0.28);
}

.badge-cache {
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(56, 189, 248, 0.28);
}

.status-dot {
    display: inline-block;
    width: 0.48rem;
    height: 0.48rem;
    border-radius: 999px;
    margin-right: 0.4rem;
    vertical-align: 0.04rem;
}

.status-live {
    background: var(--green);
    box-shadow: 0 0 5px var(--green);
}

.status-down {
    background: var(--red);
}

/* ─── Expanders & Details ─── */
details, [data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    box-shadow: var(--shadow);
    margin-bottom: 0.5rem;
}

details summary, [data-testid="stExpander"] summary {
    color: var(--text) !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] details, [data-testid="stSidebar"] [data-testid="stExpander"] {
    background: var(--sidebar-surface) !important;
    border-color: var(--line) !important;
    box-shadow: none;
}

/* ─── File Uploader ─── */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--line-strong);
    border-radius: 8px;
    padding: 16px;
    background: var(--surface);
    box-shadow: var(--shadow);
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-border);
}

/* ─── Form Inputs ─── */
[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="select"] {
    border-radius: 6px;
    background-color: var(--surface-soft) !important;
    color: var(--text) !important;
    border-color: var(--line-strong) !important;
}

code {
    background: #0d1410 !important;
    color: #6ee7b7 !important;
    border: 1px solid var(--line) !important;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.85em;
}

.model-line {
    color: var(--text);
    margin: 0.35rem 0;
    font-size: 0.88rem;
}

.citation-tag {
    display: inline-flex;
    align-items: center;
    background: var(--surface-soft);
    border: 1px solid var(--line);
    color: var(--text-muted);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.75rem;
    margin-right: 4px;
}

@media (max-width: 900px) {
    .main .block-container,
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-left: 0.85rem;
        padding-right: 0.85rem;
        padding-top: 0.75rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.8rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ─── Cached singletons ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing database...")
def get_database_manager():
    return get_db()

@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_pinecone_db():
    return PineconeDB()

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_manager():
    return EmbeddingManager()

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

    if "active_nav" not in st.session_state:
        st.session_state.active_nav = "Chat"
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
            st.session_state.active_session_id = db.create_session(title="New conversation")


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


# ─── Sidebar (Left Dashboard) ────────────────────────────────────────────────

def render_sidebar():
    db = get_database_manager()
    with st.sidebar:
        # Branding: CogniRAG in Terminal Green with short one-line description
        st.markdown(
            """
            <div class="sidebar-header">
                <div class="sidebar-title">CogniRAG</div>
                <div class="sidebar-desc">Autonomous cognitive memory & grounded document intelligence.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. New chat action (Solid high-contrast black text/icon on emerald background)
        if st.button("New chat", key="new_chat_btn", icon=":material/add_comment:", type="primary"):
            new_sid = db.create_session(title="New conversation")
            st.session_state.active_session_id = new_sid
            st.session_state.active_nav = "Chat"
            st.rerun()

        # 2. Chat history list with direct inline delete
        st.markdown("##### Conversations")
        sessions = db.list_sessions(limit=30)
        current_sid = st.session_state.get("active_session_id")

        if sessions:
            for s in sessions:
                sid = s["id"]
                stitle = s["title"]
                is_active = (sid == current_sid and st.session_state.get("active_nav", "Chat") == "Chat")

                # Truncate title cleanly
                display_title = stitle if len(stitle) <= 22 else stitle[:20] + "..."

                col_chat, col_del = st.columns([0.84, 0.16])
                with col_chat:
                    btn_key = f"chat_item_active_{sid}" if is_active else f"chat_item_inactive_{sid}"
                    if st.button(
                        display_title,
                        key=btn_key,
                        help=stitle,
                    ):
                        st.session_state.active_session_id = sid
                        st.session_state.active_nav = "Chat"
                        st.rerun()

                with col_del:
                    if st.button("", key=f"del_sess_{sid}", icon=":material/delete:", help=f"Delete '{stitle}'", type="secondary"):
                        db.delete_session(sid)
                        remaining = db.list_sessions(limit=1)
                        st.session_state.active_session_id = remaining[0]["id"] if remaining else db.create_session(title="New conversation")
                        st.rerun()

        st.divider()

        # 3. Cognitive memory summary panel
        hub: CognitiveMemoryHub = st.session_state.cognitive_hub
        mem_summary = hub.get_dashboard_summary()
        with st.expander("Memory summary", expanded=False):
            st.markdown(f"**Working memory:** `{mem_summary['working_memory']['recent_turns']}` turns")
            st.markdown(f"**Episodic recall:** `{mem_summary['episodic_memory']['total_episodes']}` episodes")
            st.markdown(f"**User preferences:** `{mem_summary['semantic_memory']['total_preferences']}` active")
            st.markdown(f"**Domain facts:** `{mem_summary['semantic_memory']['total_facts']}` learned")
            st.markdown(f"**Task recipes:** `{mem_summary['procedural_memory']['total_recipes']}` recipes")

        # 4. Quick Stats (2 metrics in 1 line)
        vdb = get_pinecone_db()
        stats = vdb.get_stats()
        cache: SemanticAnswerCache = st.session_state.cache
        cache_stats = cache.get_stats()

        col1, col2 = st.columns(2)
        col1.metric("Vectors", f"{stats['total_vector_count']:,}")
        col2.metric("Cache hits", cache_stats["hit_count"])

        st.divider()

        # 5. Bottom Navigation (Documents & Settings in 1 line like the stats box)
        col_nav_doc, col_nav_set = st.columns(2)
        curr_nav = st.session_state.get("active_nav", "Chat")

        with col_nav_doc:
            doc_is_active = (curr_nav == "Documents")
            if st.button("Documents", key="nav_docs_btn", icon=":material/description:", type="primary" if doc_is_active else "secondary"):
                st.session_state.active_nav = "Documents"
                st.rerun()

        with col_nav_set:
            set_is_active = (curr_nav == "Settings")
            if st.button("Settings", key="nav_set_btn", icon=":material/tune:", type="primary" if set_is_active else "secondary"):
                st.session_state.active_nav = "Settings"
                st.rerun()


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

    # If no messages in this session yet, display minimal empty-state helper
    if not persistent_messages:
        st.markdown(
            """
            <div class="chat-empty-state">
                <div class="chat-empty-icon">🌿</div>
                <div class="chat-empty-title">CogniRAG Assistant</div>
                <div class="chat-empty-desc">
                    Ask questions across your indexed enterprise documents with grounded citations, multi-tier memory, and verification.
                </div>
                <div class="prompt-chips">
                    <div class="prompt-chip">📌 <em>"Summarize key operational performance and milestones."</em></div>
                    <div class="prompt-chip">📊 <em>"Extract financial figures, capacity statistics, and dates."</em></div>
                    <div class="prompt-chip">🔍 <em>"Compare document policies and technical parameters."</em></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Replay message history
        for msg in persistent_messages:
            role = msg["role"]
            avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])
                _render_message_extras(msg)

    # Chat input
    query = st.chat_input("Ask a question about indexed documents...")
    if not query:
        return

    # Auto-title session if it's currently default
    active_sess = db.get_session(current_sid)
    if active_sess and active_sess["title"] in ("New Conversation", "New Chat", "New conversation", "New chat"):
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

    # Assistant response generation
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        tracer = ExecutionTracer()

        try:
            # 1. Semantic cache lookup
            cache_span = tracer.start_span("semantic_cache_lookup", component="cache")
            query_vec = emb.embed_query(query)
            cached_entry = cache.lookup(query_vec)
            tracer.finish_span(cache_span, outputs={"cache_hit": cached_entry is not None})

            if cached_entry is not None:
                sim_pct = cached_entry.get("similarity", cached_entry.get("cache_similarity", 1.0)) * 100
                st.markdown(
                    f'<span class="badge-cache">Cached Answer ({sim_pct:.1f}% match)</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(cached_entry["answer"])
                if cached_entry.get("citations"):
                    st.markdown("**Citations:** " + ", ".join([f"`{c}`" for c in cached_entry["citations"]]))

                db.save_message(
                    session_id=current_sid,
                    role="assistant",
                    content=cached_entry["answer"],
                    citations=cached_entry.get("citations", []),
                    grounding_score=1.0,
                    grounding_passed=True,
                    tokens_used=0,
                )
                tracer.finish()
                _render_execution_trace(tracer)
                return

            # 2. Query planning
            plan_span = tracer.start_span("query_planning", component="query_planner")
            plan = planner.plan(query)
            tracer.finish_span(plan_span, outputs=plan)

            # 3. Retrieve cognitive memory context
            mem_span = tracer.start_span("memory_context_build", component="cognitive_memory")
            if hasattr(hub, "build_cognitive_context"):
                cognitive_context = hub.build_cognitive_context(
                    query=query,
                    query_embedding=query_vec,
                    session_id=current_sid,
                )
            else:
                cognitive_context = hub.build_augmented_prompt_context(
                    query=query,
                    query_embedding=query_vec,
                    session_id=current_sid,
                )
            tracer.finish_span(mem_span, outputs={"memory_context_length": len(cognitive_context)})

            # 4. Stream or execute RAG pipeline
            answer_placeholder = st.empty()
            answer_parts = []
            reranked_chunks = []
            citations = []
            formatted_context = ""

            is_direct_plan = (
                plan.get("strategy") == "direct"
                or plan.get("complexity", "simple") == "simple"
                or len(plan.get("sub_queries", [])) <= 1
            )

            if is_direct_plan:
                for event_type, data in rag_chain.run_stream(
                    query=query,
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
                        answer_placeholder.markdown("".join(answer_parts))
                    elif event_type == "done":
                        citations = data["citations"]
                        reranked_chunks = data.get("reranked_chunks", reranked_chunks)
                        formatted_context = data.get("formatted_context", "")

                answer = "".join(answer_parts)
                answer_placeholder.markdown(answer)

            else:
                # Complex path: sub-queries execution and merger
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
                    with st.expander("Query decomposition"):
                        for i, sq in enumerate(merged["sub_queries"], 1):
                            st.markdown(f"**Sub-query {i}:** {sq}")

            # Citations
            if citations:
                st.markdown("**Citations:** " + ", ".join([f"`{c}`" for c in citations]))

            # Context chunks expander
            with st.expander("Retrieved context"):
                for idx, ch in enumerate(reranked_chunks):
                    st.markdown(f"**Chunk #{idx+1}:** `{ch['filename']}` page {ch['page_number']}")
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
                '<span class="badge-pass">Passed</span>'
                if eval_res["is_passed"]
                else '<span class="badge-fail">Failed</span>'
            )
            with st.expander("Grounding audit"):
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
                    st.success("No unsupported numbers detected.")

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
    with st.expander("Execution trace", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total latency", f"{summary['total_duration_ms']:,.0f} ms")
        c2.metric("LLM calls", summary["llm_calls_count"])
        c3.metric("LLM latency", f"{summary['total_llm_time_ms']:,.0f} ms")
        models_str = ", ".join([m.split("/")[-1].replace(":free", "") for m in summary["models_used"]]) or "local"
        c4.metric("Active model", models_str)

        st.markdown("##### Execution timeline")
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
        badge = '<span class="badge-pass">Passed</span>' if passed else '<span class="badge-fail">Failed</span>'
        with st.expander("Grounding audit"):
            st.markdown(
                f"**Score:** {msg['grounding_score']*100:.1f}% | Status: {badge}",
                unsafe_allow_html=True,
            )


# ─── Tab 2: Documents ─────────────────────────────────────────────────────────

def render_documents_tab():
    vdb = get_pinecone_db()
    emb = get_embedding_manager()

    col_title, col_btn = st.columns([0.82, 0.18])
    with col_title:
        st.markdown("### Upload documents")
        st.caption("Index PDF, Word, Excel, CSV, and text files for vector retrieval.")
    with col_btn:
        if st.button("← Back to Chat", key="back_to_chat_from_docs", use_container_width=True):
            st.session_state.active_nav = "Chat"
            st.rerun()

    uploaded = st.file_uploader(
        "Files to index",
        type=["pdf", "csv", "xlsx", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and st.button("Index files", type="primary"):
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
                progress.progress((i + 1) / len(uploaded), text=f"Indexed {f.name}: {n} chunks")
            except Exception as exc:
                has_error = True
                diag = ErrorDiagnosticManager.diagnose(exc, context=f"Uploading and Indexing Document `{f.name}`")
                st.markdown(diag.format_markdown(), unsafe_allow_html=True)

        progress.empty()
        if not has_error:
            st.success(f"Successfully indexed {len(uploaded)} file(s): {total_chunks} total chunks.")
            st.rerun()
        else:
            st.warning(f"Indexing completed with issues. Total indexed chunks: {total_chunks}")

    st.divider()

    # Document registry
    st.markdown("### Indexed documents")
    registry = vdb.list_documents()

    if not registry:
        st.info("No documents indexed yet. Upload files above to begin.")
        return

    stats = vdb.get_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Vectors", f"{stats['total_vector_count']:,}")
    col2.metric("Documents", stats.get("db_documents", len(registry)))
    col3.metric("Chunks", f"{stats.get('db_chunks', 0):,}")

    st.markdown("")

    for filename, info in registry.items():
        col_name, col_chunks, col_date, col_del = st.columns([4, 1.5, 2.5, 1])
        col_name.markdown(f"`{filename}`")
        col_chunks.markdown(f"**{info['chunk_count']}** chunks")
        indexed_at = info.get("indexed_at", "Not available")[:10]
        col_date.caption(f"Indexed: {indexed_at}")
        if col_del.button("", key=f"del_doc_{filename}", icon=":material/delete:", help=f"Delete {filename}", type="secondary"):
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

    col_title, col_btn = st.columns([0.82, 0.18])
    with col_title:
        st.markdown("### System & Memory Settings")
        st.caption("Inspect cognitive tiers, model configurations, and memory graph.")
    with col_btn:
        if st.button("← Back to Chat", key="back_to_chat_from_set", use_container_width=True):
            st.session_state.active_nav = "Chat"
            st.rerun()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Memory")
        
        # User Preferences Editor
        with st.expander("User preferences", expanded=True):
            prefs = hub.semantic_memory.get_all_preferences()
            for k, v in prefs.items():
                col_k, col_v, col_d = st.columns([2.5, 4, 1])
                col_k.markdown(f"**{k}**")
                col_v.caption(v)
                if col_d.button("", key=f"del_pref_{k}", icon=":material/cancel:", help=f"Erase preference '{k}'", type="secondary"):
                    hub.semantic_memory.delete_preference(k)
                    st.rerun()

            st.markdown("##### Add or update preference")
            col_nk, col_nv = st.columns(2)
            new_k = col_nk.text_input("Key", placeholder="e.g. response_tone", key="new_pref_k")
            new_v = col_nv.text_input("Value", placeholder="e.g. Executive & concise", key="new_pref_v")
            if st.button("Save preference") and new_k and new_v:
                hub.semantic_memory.set_preference(new_k, new_v)
                st.success(f"Saved preference `{new_k}`")
                st.rerun()

        # Domain Facts Graph
        with st.expander("Domain facts", expanded=False):
            facts = hub.semantic_memory.get_all_facts(limit=20)
            if facts:
                for f in facts:
                    col_f, col_fd = st.columns([6, 1])
                    col_f.markdown(f"- **{f['subject']}**: *{f['predicate']}* `{f['object']}`")
                    if col_fd.button("", key=f"del_fact_{f['id']}", icon=":material/cancel:", help="Erase domain fact", type="secondary"):
                        hub.semantic_memory.delete_fact(f["id"])
                        st.rerun()
            else:
                st.caption("No domain facts stored yet. Learned facts will appear here.")

            st.markdown("##### Add domain fact")
            col_s, col_p, col_o = st.columns(3)
            subj = col_s.text_input("Subject", placeholder="NTPC", key="fact_s")
            pred = col_p.text_input("Predicate", placeholder="commercial_capacity", key="fact_p")
            obj = col_o.text_input("Object", placeholder="76 GW", key="fact_o")
            if st.button("Add Fact") and subj and pred and obj:
                hub.semantic_memory.add_fact(subj, pred, obj, source="manual_entry")
                st.success("Domain fact added.")
                st.rerun()

        # Procedural Task Recipes Catalog
        with st.expander("Task recipes", expanded=False):
            recipes = hub.procedural_memory.get_all_recipes()
            for r in recipes:
                st.markdown(f"**Recipe:** `{r['name']}`")
                st.caption("Triggers: " + ", ".join([f"`{t}`" for t in r.get("trigger_patterns", [])]))
                for step in r.get("steps", []):
                    st.markdown(f"&nbsp;&nbsp;{step}")
                st.divider()

        # Episodic History
        with st.expander("Past answers", expanded=False):
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

        st.markdown("### Retrieval settings")
        st.session_state.retrieval_top_k = st.slider(
            "Initial retrieval top-K", 5, 30, st.session_state.retrieval_top_k
        )
        st.session_state.rerank_top_k = st.slider(
            "After reranking top-K", 2, 10, st.session_state.rerank_top_k
        )

    with col_right:
        st.markdown("### Model status")
        active_models = llm.get_active_models()
        for task, model in active_models.items():
            if model:
                friendly_name = format_model_name(model)
                st.markdown(
                    f'<div class="model-line"><span class="status-dot status-live"></span>'
                    f'<strong>{task}</strong>: <code>{friendly_name}</code></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="model-line"><span class="status-dot status-down"></span>'
                    f'<strong>{task}</strong>: offline</div>',
                    unsafe_allow_html=True,
                )
        if not llm.is_available():
            st.warning("OpenRouter is unavailable. All requests will use Gemini as fallback.")
        else:
            st.success(f"OpenRouter connected: {len(llm._live_free_models)} free models available.")

        st.divider()

        st.markdown("### Database and memory")
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("Clear answer cache", key="clear_cache_btn"):
            cache.clear()
            st.success("Answer cache cleared.")
            st.rerun()

        if col_c2.button("Reset cognitive memory", key="reset_memory_btn"):
            hub.clear_all()
            st.success("Cognitive memory reset.")
            st.rerun()

        st.caption(f"SQLite Database Path: `{DB_PATH}`")

        st.divider()

        st.markdown("### API keys")
        from config import PINECONE_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY
        def mask(k): return k[:8] + "..." + k[-4:] if len(k) > 12 else "not set"
        st.code(f"PINECONE_API_KEY   = {mask(PINECONE_API_KEY)}")
        st.code(f"OPENROUTER_API_KEY = {mask(OPENROUTER_API_KEY)}")
        st.code(f"GEMINI_API_KEY     = {mask(GEMINI_API_KEY)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    init_session_state()
    render_sidebar()

    active_nav = st.session_state.get("active_nav", "Chat")

    # Render view based on active navigation
    if active_nav == "Documents":
        render_documents_tab()
    elif active_nav == "Settings":
        render_settings_tab()
    else:
        render_chat_tab()


if __name__ == "__main__":
    main()
