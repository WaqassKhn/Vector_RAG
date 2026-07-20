import os
import sys
import importlib
import pandas as pd
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Ensure local modules can be imported
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
import rag.llm
import rag.chain
import evaluation.grounding_eval

importlib.reload(config)
importlib.reload(rag.llm)
importlib.reload(rag.chain)
importlib.reload(evaluation.grounding_eval)

from config import (
    GEMINI_API_KEY, DEFAULT_GEMINI_MODEL, DEFAULT_LOCAL_EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, INITIAL_TOP_K, RERANKED_TOP_K,
    UPLOADS_DIR, VECTOR_DB_DIR
)
from pipeline.parser import DocumentParser
from pipeline.cleaner import DocumentCleaner
from pipeline.chunker import DocumentChunker
from vectorstore.embeddings import EmbeddingManager
from vectorstore.vector_db import VectorDatabase
from rag.reranker import HybridReranker
from rag.llm import GeminiLLM
from rag.chain import RAGChain
from evaluation.grounding_eval import DocumentGroundingEvaluator

# Load environment variables
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="Grounded RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom Glassmorphic ChatGPT-Style CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 20px;
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid #30363d;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .floating-island {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .badge-pass {
        background-color: #238636;
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-fail {
        background-color: #da3633;
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def auto_index_uploads(vdb: VectorDatabase, embedding_mgr: EmbeddingManager, chunk_size: int, chunk_overlap: int):
    """Auto-indexes files placed inside data/uploads/ on startup or when new files are added."""
    upload_files = [p for p in UPLOADS_DIR.iterdir() if p.is_file()]
    
    # If no files in uploads, copy sample dummy report
    if not upload_files:
        sample_file = BASE_DIR / "sample_data" / "dirty_company_report.txt"
        if sample_file.exists():
            dest = UPLOADS_DIR / "dirty_company_report.txt"
            dest.write_text(sample_file.read_text(encoding="utf-8"), encoding="utf-8")
            upload_files = [dest]

    # Find files that are not yet indexed in vdb
    indexed_filenames = set(c.get("filename") for c in vdb.chunks_metadata) if vdb.chunks_metadata else set()
    unindexed_files = [f for f in upload_files if f.name not in indexed_filenames]

    if unindexed_files:
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        new_chunks = []
        for fpath in unindexed_files:
            parsed = DocumentParser.parse_file(str(fpath))
            chunks = chunker.chunk_parsed_document(parsed)
            new_chunks.extend(chunks)

        if new_chunks and embedding_mgr:
            texts = [c["text"] for c in new_chunks]
            vecs = embedding_mgr.embed_texts(texts)
            vdb.add_chunks(new_chunks, vecs)
            vdb.save_index()


def main():
    # Session State Initialization
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = VectorDatabase(VECTOR_DB_DIR)
        st.session_state.vector_db.load_index()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Floating Island Settings Control
    with st.expander("⚙️ System Parameters & Document Indexing (Floating Island)", expanded=False):
        col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
        
        with col_s1:
            st.markdown("#### 🔑 API & Embedding Settings")
            api_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
            use_gemini_emb = st.checkbox("Use Gemini Embeddings API", value=False)
        
        with col_s2:
            st.markdown("#### 🎛️ Retrieval Parameters")
            chunk_size = st.slider("Chunk Size", 200, 1500, DEFAULT_CHUNK_SIZE, 50)
            chunk_overlap = st.slider("Chunk Overlap", 50, 400, DEFAULT_CHUNK_OVERLAP, 10)
            initial_k = st.slider("Initial Top-K Candidates", 5, 30, INITIAL_TOP_K)
            rerank_k = st.slider("Reranked Top-K", 2, 10, RERANKED_TOP_K)

        with col_s3:
            st.markdown("#### 📁 Local Folder Indexing")
            st.caption(f"Folder: `data/uploads/`")
            uploaded_files = st.file_uploader("Upload additional PDFs/documents", type=["pdf", "csv", "xlsx", "docx", "txt"], accept_multiple_files=True)
            
            if st.button("🚀 Re-index All Documents in `data/uploads/`"):
                if uploaded_files:
                    for f in uploaded_files:
                        save_p = UPLOADS_DIR / f.name
                        with open(save_p, "wb") as w:
                            w.write(f.getbuffer())
                
                # Re-index
                st.session_state.vector_db.clear()
                embedding_mgr = EmbeddingManager(use_gemini=use_gemini_emb, api_key=api_key)
                auto_index_uploads(st.session_state.vector_db, embedding_mgr, chunk_size, chunk_overlap)
                st.success("Re-indexed all documents successfully!")
                st.rerun()

            if st.button("🧹 Clear Index"):
                st.session_state.vector_db.clear()
                st.session_state.messages = []
                st.success("Cleared vector database.")
                st.rerun()

    # Shared Dependencies
    api_key_val = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    embedding_mgr = EmbeddingManager(use_gemini=False, api_key=api_key_val)
    llm = GeminiLLM(model_name=DEFAULT_GEMINI_MODEL, api_key=api_key_val)
    reranker = HybridReranker()

    # Auto-index data/uploads on initial startup
    auto_index_uploads(st.session_state.vector_db, embedding_mgr, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

    rag_chain = RAGChain(
        vector_db=st.session_state.vector_db,
        embedding_manager=embedding_mgr,
        llm=llm,
        reranker=reranker
    )
    evaluator = DocumentGroundingEvaluator(llm=llm)

    # Top Header
    num_chunks = len(st.session_state.vector_db.chunks_metadata)
    st.markdown(f"""
    <div class="chat-header">
        <div>
            <h2 style="margin:0; font-size: 1.4rem;">🤖 Grounded RAG Assistant</h2>
            <span style="color: #8b949e; font-size: 0.85rem;">Document Index: {num_chunks} chunks stored in <code>data/vector_db/</code></span>
        </div>
        <div>
            <span style="color: #58a6ff; font-weight: 600;">Model: Gemini Flash</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display Chat Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg.get("citations"):
                st.markdown("**Citations:** " + ", ".join([f"`{c}`" for c in msg["citations"]]))
            
            if msg.get("chunks"):
                with st.expander("📚 Retrieved Context Chunks"):
                    for idx, ch in enumerate(msg["chunks"]):
                        st.markdown(f"**Chunk #{idx+1} [{ch['filename']} Page {ch['page_number']}]**")
                        st.code(ch["text"], language="text")

            if msg.get("eval_res"):
                ev = msg["eval_res"]
                badge_html = f'<span class="badge-pass">PASSED</span>' if ev["is_passed"] else f'<span class="badge-fail">FAILED</span>'
                with st.expander("🧪 Document Grounding Audit"):
                    st.markdown(f"**Grounding Score:** {ev['overall_grounding_score']*100:.1f}% | **Faithfulness:** {ev['faithfulness_score']*100:.1f}% | Status: {badge_html}", unsafe_allow_html=True)
                    if ev["unsupported_numbers"]:
                        st.error("Unsupported Numbers Detected: " + ", ".join(ev["unsupported_numbers"]))
                    else:
                        st.success("Zero numerical hallucinations detected.")

    # Bottom Text Box Input (ChatGPT Style)
    user_query = st.chat_input("Ask a question about your documents stored in data/uploads/...")

    if user_query:
        # Display User Input
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate RAG Response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context & generating grounded answer..."):
                rag_res = rag_chain.run(query=user_query, initial_top_k=INITIAL_TOP_K, rerank_top_k=RERANKED_TOP_K)
                
                # Perform Grounding Audit
                eval_res = evaluator.evaluate_grounding(
                    query=user_query,
                    answer=rag_res["answer"],
                    retrieved_chunks=rag_res["reranked_chunks"]
                )

                st.markdown(rag_res["answer"])
                
                if rag_res["citations"]:
                    st.markdown("**Citations:** " + ", ".join([f"`{c}`" for c in rag_res["citations"]]))

                with st.expander("📚 Retrieved Context Chunks"):
                    for idx, ch in enumerate(rag_res["reranked_chunks"]):
                        st.markdown(f"**Chunk #{idx+1} [{ch['filename']} Page {ch['page_number']}]**")
                        st.code(ch["text"], language="text")

                badge_html = f'<span class="badge-pass">PASSED</span>' if eval_res["is_passed"] else f'<span class="badge-fail">FAILED</span>'
                with st.expander("🧪 Document Grounding Audit"):
                    st.markdown(f"**Grounding Score:** {eval_res['overall_grounding_score']*100:.1f}% | **Faithfulness:** {eval_res['faithfulness_score']*100:.1f}% | Status: {badge_html}", unsafe_allow_html=True)
                    if eval_res["unsupported_numbers"]:
                        st.error("Unsupported Numbers Detected: " + ", ".join(eval_res["unsupported_numbers"]))
                    else:
                        st.success("Zero numerical hallucinations detected.")

                # Store Assistant Message in Session State
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": rag_res["answer"],
                    "citations": rag_res["citations"],
                    "chunks": rag_res["reranked_chunks"],
                    "eval_res": eval_res
                })


if __name__ == "__main__":
    main()
