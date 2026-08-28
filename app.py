"""
app.py — RAG_NTPC Production Application with Persistent Multi-Session Memory
─────────────────────────────────────────────────────────────────────────────
Features:
  - Persistent SQLite Database (data/rag_app.db): Zero data loss on interface close or reload.
  - Multi-Session Chat Manager: Create, switch, rename, and delete conversation threads.
  - Multi-Tier Cognitive Memory Engine: Working, Episodic, Semantic, and Procedural memory.
  - 3-Tab Streamlit UI:
      Tab 1: 💬 Chat      — streaming RAG chat with grounding audit, citations, and session history
      Tab 2: 📁 Documents — upload, index, and manage documents with 0 API token cost
      Tab 3: ⚙️ Settings  — cognitive memory explorer, model status, token budget, database inspector
  - Sidebar: Session selector + token quota meter + cognitive memory summary + active models.
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

from config import (
    VECTOR_DB_DIR, UPLOADS_DIR, DB_PATH,
    INITIAL_TOP_K, RERANKED_TOP_K,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    GEMINI_PRO_DAILY_REQUEST_CAP, GEMINI_FLASH_DAILY_REQUEST_CAP,
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
from rag.token_counter import TokenTracker
from evaluation.grounding_eval import DocumentGroundingEvaluator

logging.basicConfig(level=logging.WARNING)

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Assistant — Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(22, 27, 34, 0.95);
    border-right: 1px solid #30363d;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(22, 27, 34, 0.8);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #8b949e;
    font-weight: 500;
    padding: 8px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: rgba(88, 166, 255, 0.15) !important;
    color: #58a6ff !important;
    border: 1px solid rgba(88, 166, 255, 0.3) !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: rgba(22, 27, 34, 0.6);
    border: 1px solid #30363d;
    border-radius: 12px;
    margin-bottom: 8px;
    backdrop-filter: blur(10px);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(22, 27, 34, 0.8);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(35, 134, 54, 0.4);
}

/* Secondary / Danger buttons */
button[kind="secondary"] {
    background: rgba(218, 54, 51, 0.15) !important;
    color: #f85149 !important;
    border: 1px solid rgba(218, 54, 51, 0.3) !important;
}

/* Badges */
.badge-pass {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-fail {
    background: linear-gradient(135deg, #da3633, #f85149);
    color: white;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-cache {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-token {
    background: rgba(110, 118, 129, 0.2);
    color: #58a6ff;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(88, 166, 255, 0.3);
}

/* Expanders */
details {
    background: rgba(22, 27, 34, 0.5);
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 4px 12px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 20px;
    background: rgba(22, 27, 34, 0.3);
}

/* Code blocks */
code {
    background: rgba(110, 118, 129, 0.15);
    color: #e6edf3;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.85em;
}

/* Model status dots */
.model-live { color: #3fb950; font-size: 0.8rem; }
.model-down { color: #f85149; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ─── Cached singletons ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing persistent database...")
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
    if "token_tracker" not in st.session_state:
        st.session_state.token_tracker = TokenTracker()
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

def index_file(fpath: Path, vdb: PineconeDB, emb: EmbeddingManager, tracker: TokenTracker) -> int:
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
    
    # Record local tokens saved (0 API cost)
    tracker.record_local_embedding(texts)
    return len(chunks)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    db = get_database_manager()
    with st.sidebar:
        st.markdown("## 🤖 RAG Assistant")
        st.caption("Persistent Document Intelligence Engine")
        st.divider()

        # 1. Chat Sessions Manager
        st.markdown("### 💬 Conversations")
        sessions = db.list_sessions(limit=30)
        current_sid = st.session_state.get("active_session_id")

        if st.button("➕ New Conversation", use_container_width=True, type="primary"):
            new_sid = db.create_session(title="New Conversation")
            st.session_state.active_session_id = new_sid
            st.rerun()

        if sessions:
            session_titles = {s["id"]: s["title"] for s in sessions}
            session_ids = list(session_titles.keys())
            current_index = session_ids.index(current_sid) if current_sid in session_ids else 0

            selected_sid = st.selectbox(
                "Select Conversation",
                options=session_ids,
                format_func=lambda sid: f"📝 {session_titles[sid]}",
                index=current_index,
                label_visibility="collapsed",
            )

            if selected_sid != current_sid:
                st.session_state.active_session_id = selected_sid
                st.rerun()

            # Session rename & delete controls
            with st.expander("⚙️ Manage Active Chat", expanded=False):
                active_sess = db.get_session(selected_sid)
                if active_sess:
                    new_title = st.text_input("Rename title", value=active_sess["title"])
                    col_ren, col_del = st.columns(2)
                    if col_ren.button("Save Title", use_container_width=True):
                        db.update_session_title(selected_sid, new_title)
                        st.rerun()
                    if col_del.button("🗑️ Delete", kind="secondary", use_container_width=True):
                        db.delete_session(selected_sid)
                        remaining = db.list_sessions(limit=1)
                        st.session_state.active_session_id = remaining[0]["id"] if remaining else db.create_session()
                        st.rerun()

        st.divider()

        # 2. Token Budget & Daily Quota Meter
        tracker: TokenTracker = st.session_state.token_tracker
        t_summary = tracker.get_summary()
        with st.expander("📊 Daily Token Budget & Quota", expanded=True):
            col_t1, col_t2 = st.columns(2)
            col_t1.metric("Tokens Used", f"{t_summary['total_tokens']:,}")
            col_t2.metric("Total Cost", "$0.00", help="100% Free Tier Architecture")

            st.caption(f"**Gemini Pro Quota (50 RPD Cap):** {t_summary['pro_quota_used_pct']}% used")
            st.progress(t_summary['pro_quota_used_pct'] / 100.0)

            st.caption(f"**Gemini Flash Quota (1500 RPD Cap):** {t_summary['flash_quota_used_pct']}% used")
            st.progress(t_summary['flash_quota_used_pct'] / 100.0)

            st.caption(f"💾 **Tokens Saved (Local + Cache):** {t_summary['saved_local_embed_tokens'] + t_summary['saved_cache_tokens']:,}")

        st.divider()

        # 3. Cognitive Memory Summary Panel
        hub: CognitiveMemoryHub = st.session_state.cognitive_hub
        mem_summary = hub.get_dashboard_summary()
        with st.expander("🧠 Cognitive Memory Hub", expanded=False):
            st.markdown(f"**Working Memory:** `{mem_summary['working_memory']['recent_turns']}` recent turns")
            st.markdown(f"**Episodic Recall:** `{mem_summary['episodic_memory']['total_episodes']}` past episodes")
            st.markdown(f"**User Preferences:** `{mem_summary['semantic_memory']['total_preferences']}` active")
            st.markdown(f"**Domain Facts:** `{mem_summary['semantic_memory']['total_facts']}` learned")
            st.markdown(f"**Task Recipes:** `{mem_summary['procedural_memory']['total_recipes']}` available")

        st.divider()

        # 4. Active model status
        llm = get_openrouter_llm()
        with st.expander("🔀 Active Models", expanded=False):
            active = llm.get_active_models()
            for task, model in active.items():
                if model:
                    short = model.split("/")[-1].replace(":free", "")
                    st.markdown(
                        f'<span class="model-live">●</span> **{task}**: `{short}`',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span class="model-down">●</span> **{task}**: no model available',
                        unsafe_allow_html=True,
                    )

        st.divider()

        # 5. Quick stats
        vdb = get_pinecone_db()
        stats = vdb.get_stats()
        cache: SemanticAnswerCache = st.session_state.cache
        cache_stats = cache.get_stats()

        col1, col2 = st.columns(2)
        col1.metric("Vectors", f"{stats['total_vector_count']:,}")
        col2.metric("Cache hits", cache_stats["hit_count"])


# ─── Tab 1: Chat ──────────────────────────────────────────────────────────────

def render_chat_tab():
    db = get_database_manager()
    vdb = get_pinecone_db()
    emb = get_embedding_manager()
    llm = get_openrouter_llm()
    reranker = get_reranker()
    hub: CognitiveMemoryHub = st.session_state.cognitive_hub
    cache: SemanticAnswerCache = st.session_state.cache
    tracker: TokenTracker = st.session_state.token_tracker
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
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_message_extras(msg)

    # Chat input
    query = st.chat_input("Ask a question about your documents...")
    if not query:
        return

    # Auto-title session if it's currently default
    active_sess = db.get_session(current_sid)
    if active_sess and active_sess["title"] in ("New Conversation", "New Chat"):
        new_title = query[:32] + ("..." if len(query) > 32 else "")
        db.update_session_title(current_sid, new_title)

    # Save and display user message in DB
    db.save_message(
        session_id=current_sid,
        role="user",
        content=query,
    )
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # 1. Embed query
        query_vec = emb.embed_query(query)

        # 2. Check semantic cache
        cache_hit = cache.lookup(query_vec)
        if cache_hit:
            st.markdown(cache_hit["answer"])
            st.markdown(
                f'<span class="badge-cache">⚡ Cache Hit ({cache_hit["cache_similarity"]:.2%} similarity)</span> '
                f'<span class="badge-token">0 API Tokens Consumed</span>',
                unsafe_allow_html=True,
            )
            tracker.record_cache_hit(query, cache_hit["answer"])

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
            # Update cognitive memory
            hub.post_interaction_update(
                query=query,
                answer=cache_hit["answer"],
                citations=cache_hit["citations"],
                query_embedding=query_vec,
                session_id=current_sid,
            )
            return

        # 3. Assemble Cognitive Memory Context (Working, Episodic, Semantic, Procedural)
        cognitive_context = hub.build_cognitive_context(
            query=query,
            query_embedding=query_vec,
            session_id=current_sid,
        )

        # 4. Plan the query (incorporating procedural recipe guidance)
        matched_recipe = hub.procedural_memory.match_recipe(query)
        procedural_hint = hub.procedural_memory.get_context_string(query) if matched_recipe else None

        with st.spinner("🧠 Planning retrieval strategy..."):
            plan = planner.plan(query, procedural_hint=procedural_hint)
            tracker.record_query(query, json.dumps(plan), task="decompose")

        # 5. Execute retrieval + generation
        formatted_context = ""

        if plan["complexity"] == "simple" or len(plan["sub_queries"]) == 1:
            # ── Simple path: stream directly ──
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

            # Record generation token usage
            t_delta = tracker.record_query(query, answer, context=formatted_context, task="answer")

        else:
            # ── Complex path: parallel sub-queries → merge ──
            with st.spinner(f"🔍 Running {len(plan['sub_queries'])} retrieval passes..."):
                sub_results = []
                for sub_q in plan["sub_queries"]:
                    result = rag_chain.run(
                        query=sub_q,
                        initial_top_k=st.session_state.retrieval_top_k,
                        rerank_top_k=st.session_state.rerank_top_k,
                        filter_filenames=plan.get("doc_scope"),
                        memory_context=cognitive_context,
                    )
                    sub_results.append(result)

            with st.spinner("🔗 Synthesizing multi-document answer..."):
                merged = merger.merge(original_query=query, sub_results=sub_results)

            answer = merged["answer"]
            reranked_chunks = merged["reranked_chunks"]
            citations = merged["citations"]
            formatted_context = merged.get("formatted_context", "")
            st.markdown(answer)

            t_delta = tracker.record_query(query, answer, context=formatted_context, task="answer")

            if merged.get("sub_queries"):
                with st.expander("🔀 Query Decomposition"):
                    for i, sq in enumerate(merged["sub_queries"], 1):
                        st.markdown(f"**Sub-query {i}:** {sq}")

        # Citations
        if citations:
            st.markdown("**Citations:** " + ", ".join([f"`{c}`" for c in citations]))

        # Context chunks expander
        with st.expander("📚 Retrieved Context Chunks"):
            for idx, ch in enumerate(reranked_chunks):
                st.markdown(f"**Chunk #{idx+1}** — `{ch['filename']}` Page {ch['page_number']}")
                st.code(ch["text"], language="text")

        # Grounding audit
        with st.spinner("🧪 Running grounding audit..."):
            eval_res = evaluator.evaluate_grounding(
                query=query,
                answer=answer,
                retrieved_chunks=reranked_chunks,
            )
            tracker.record_query(query, json.dumps(eval_res), context=formatted_context, task="judge")

        badge = (
            '<span class="badge-pass">✓ PASSED</span>'
            if eval_res["is_passed"]
            else '<span class="badge-fail">✗ FAILED</span>'
        )
        token_badge = f'<span class="badge-token">~{t_delta["total_tokens"]} Tokens</span>'
        with st.expander("🧪 Grounding Audit"):
            st.markdown(
                f"**Score:** {eval_res['overall_grounding_score']*100:.1f}% | "
                f"**Faithfulness:** {eval_res['faithfulness_score']*100:.1f}% | "
                f"**Numerical:** {eval_res['numerical_accuracy_score']*100:.1f}% | "
                f"Status: {badge} | Cost: {token_badge}",
                unsafe_allow_html=True,
            )
            if eval_res["unsupported_numbers"]:
                st.error("⚠️ Unsupported numbers: " + ", ".join(eval_res["unsupported_numbers"]))
            else:
                st.success("✓ Zero numerical hallucinations detected.")

        # Store in cache
        cache.store(query_vec, answer, citations, query_text=query)

        # Update Cognitive Memory Hub (Working turn, Episodic record, Semantic facts)
        hub.post_interaction_update(
            query=query,
            answer=answer,
            citations=citations,
            query_embedding=query_vec,
            session_id=current_sid,
            prompt_tokens=t_delta.get("prompt_tokens", 0),
            completion_tokens=t_delta.get("completion_tokens", 0),
            model=getattr(llm, "model", "openrouter"),
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
            tokens_used=t_delta["total_tokens"],
        )


def _render_message_extras(msg: dict):
    """Renders citation/eval extras for replayed messages from database."""
    if msg.get("citations") and len(msg["citations"]) > 0:
        st.markdown("**Citations:** " + ", ".join([f"`{c}`" for c in msg["citations"]]))

    if msg.get("grounding_score", 0.0) > 0:
        passed = msg.get("grounding_passed", False)
        badge = '<span class="badge-pass">✓ PASSED</span>' if passed else '<span class="badge-fail">✗ FAILED</span>'
        token_badge = f'<span class="badge-token">~{msg.get("tokens_used", 0)} Tokens</span>'
        with st.expander("🧪 Grounding Audit"):
            st.markdown(
                f"**Score:** {msg['grounding_score']*100:.1f}% | Status: {badge} {token_badge}",
                unsafe_allow_html=True,
            )


# ─── Tab 2: Documents ─────────────────────────────────────────────────────────

def render_documents_tab():
    vdb = get_pinecone_db()
    emb = get_embedding_manager()
    tracker: TokenTracker = st.session_state.token_tracker

    st.markdown("### 📤 Upload Documents")
    st.caption("ℹ️ Document ingestion and vector indexing use local embeddings (`MiniLM-L6-v2`) and cost **0 API tokens**.")
    uploaded = st.file_uploader(
        "Drag & drop files here",
        type=["pdf", "csv", "xlsx", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and st.button("🚀 Index Uploaded Files", type="primary"):
        progress = st.progress(0, text="Starting indexing...")
        total_chunks = 0
        for i, f in enumerate(uploaded):
            save_path = UPLOADS_DIR / f.name
            with open(save_path, "wb") as w:
                w.write(f.getbuffer())
            progress.progress((i + 0.5) / len(uploaded), text=f"Parsing {f.name}...")
            try:
                n = index_file(save_path, vdb, emb, tracker)
                total_chunks += n
                progress.progress((i + 1) / len(uploaded), text=f"Indexed {f.name} → {n} chunks")
            except Exception as exc:
                st.error(f"Failed to index {f.name}: {exc}")
        progress.empty()
        st.success(f"✓ Indexed {len(uploaded)} file(s) → {total_chunks} total chunks saved to persistent DB at $0.00 cost.")
        st.rerun()

    st.divider()

    # Document registry
    st.markdown("### 📋 Persistent Document Registry")
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
        col_name.markdown(f"📄 `{filename}`")
        col_chunks.markdown(f"**{info['chunk_count']}** chunks")
        indexed_at = info.get("indexed_at", "—")[:10]
        col_date.caption(f"Indexed: {indexed_at}")
        if col_del.button("🗑️", key=f"del_{filename}", help=f"Delete {filename}"):
            deleted = vdb.delete_by_filename(filename)
            st.success(f"Deleted {deleted} chunks for `{filename}`.")
            st.rerun()


# ─── Tab 3: Settings & Cognitive Memory Explorer ──────────────────────────────

def render_settings_tab():
    db = get_database_manager()
    llm = get_openrouter_llm()
    hub: CognitiveMemoryHub = st.session_state.cognitive_hub
    cache: SemanticAnswerCache = st.session_state.cache
    tracker: TokenTracker = st.session_state.token_tracker

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🧠 Cognitive Memory Explorer")
        
        # User Preferences Editor
        with st.expander("⚙️ User Preferences (Semantic Memory)", expanded=True):
            prefs = hub.semantic_memory.get_all_preferences()
            for k, v in prefs.items():
                col_k, col_v, col_d = st.columns([2.5, 4, 1])
                col_k.markdown(f"**{k}**")
                col_v.caption(v)
                if col_d.button("✕", key=f"del_pref_{k}"):
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
        with st.expander("📚 Domain Facts Graph (Semantic Memory)", expanded=False):
            facts = hub.semantic_memory.get_all_facts(limit=20)
            if facts:
                for f in facts:
                    col_f, col_fd = st.columns([6, 1])
                    col_f.markdown(f"• **{f['subject']}** — *{f['predicate']}*: `{f['object']}`")
                    if col_fd.button("✕", key=f"del_fact_{f['id']}"):
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
        with st.expander("🛠️ Task Execution Recipes (Procedural Memory)", expanded=False):
            recipes = hub.procedural_memory.get_all_recipes()
            for r in recipes:
                st.markdown(f"**Recipe:** `{r['name']}`")
                st.caption("Triggers: " + ", ".join([f"`{t}`" for t in r.get("trigger_patterns", [])]))
                for step in r.get("steps", []):
                    st.markdown(f"&nbsp;&nbsp;{step}")
                st.divider()

        # Episodic History
        with st.expander("🕒 Episodic Memory History", expanded=False):
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

        st.markdown("### 🎛️ Retrieval Parameters")
        st.session_state.retrieval_top_k = st.slider(
            "Initial retrieval top-K", 5, 30, st.session_state.retrieval_top_k
        )
        st.session_state.rerank_top_k = st.slider(
            "After reranking top-K", 2, 10, st.session_state.rerank_top_k
        )

    with col_right:
        st.markdown("### 🔀 OpenRouter Model Status")
        active_models = llm.get_active_models()
        for task, model in active_models.items():
            if model:
                st.markdown(
                    f'<span class="model-live">●</span> **{task}** → `{model}`',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="model-down">●</span> **{task}** → no live model',
                    unsafe_allow_html=True,
                )
        if not llm.is_available():
            st.warning("⚠️ OpenRouter is unavailable. All requests will use Gemini as fallback.")
        else:
            st.success(f"✓ OpenRouter connected — {len(llm._live_free_models)} free models live.")

        st.divider()

        st.markdown("### 📊 Token Budget & Quota Stats")
        t_stats = tracker.get_summary()
        col_1, col_2, col_3 = st.columns(3)
        col_1.metric("Input Tokens", f"{t_stats['total_input_tokens']:,}")
        col_2.metric("Output Tokens", f"{t_stats['total_output_tokens']:,}")
        col_3.metric("Total Tokens", f"{t_stats['total_tokens']:,}")

        st.markdown(f"**Gemini Pro Daily Usage:** `{t_stats['total_requests']}` / {GEMINI_PRO_DAILY_REQUEST_CAP} requests ({t_stats['pro_quota_used_pct']}%)")
        st.markdown(f"**Gemini Flash Daily Usage:** `{t_stats['total_requests']}` / {GEMINI_FLASH_DAILY_REQUEST_CAP} requests ({t_stats['flash_quota_used_pct']}%)")
        st.markdown(f"**Tokens Saved Locally:** `{t_stats['saved_local_embed_tokens'] + t_stats['saved_cache_tokens']:,}` tokens")

        st.divider()

        st.markdown("### 🗄️ Database & Memory Controls")
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("🧹 Clear Answer Cache", use_container_width=True):
            cache.clear()
            st.success("Answer cache cleared.")
            st.rerun()

        if col_c2.button("🧠 Reset Cognitive Memory", use_container_width=True):
            hub.clear_all()
            st.success("Cognitive memory reset.")
            st.rerun()

        st.caption(f"💾 **SQLite Database Path:** `{DB_PATH}`")

        st.divider()

        st.markdown("### 🔑 API Keys (masked)")
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
        "<h1 style='margin-bottom:4px'>🤖 RAG Assistant</h1>"
        "<p style='color:#8b949e;margin-top:0'>Persistent Document Intelligence — Powered by Pinecone + OpenRouter + SQLite</p>",
        unsafe_allow_html=True,
    )

    tab_chat, tab_docs, tab_settings = st.tabs(["💬 Chat", "📁 Documents", "⚙️ Settings"])

    with tab_chat:
        render_chat_tab()

    with tab_docs:
        render_documents_tab()

    with tab_settings:
        render_settings_tab()


if __name__ == "__main__":
    main()
