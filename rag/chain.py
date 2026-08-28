"""
rag/chain.py
────────────
End-to-end RAG chain: embed → retrieve → rerank → generate.

Supports both Pinecone (production) and FAISS VectorDatabase (eval harness)
via duck-typing — both expose .search() and .chunks_metadata.

Two execution modes:
  run()        — returns a complete dict (used by MergeAgent, eval harness, cache)
  run_stream() — yields (type, data) tuples for token-by-token streaming in UI
"""

import re
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from config import INITIAL_TOP_K, RERANKED_TOP_K
from vectorstore.embeddings import EmbeddingManager
from rag.reranker import HybridReranker

# Accept either VectorDatabase (eval) or PineconeDB (production) via duck-typing
try:
    from vectorstore.pinecone_db import PineconeDB
except ImportError:
    PineconeDB = None

try:
    from vectorstore.vector_db import VectorDatabase
except ImportError:
    VectorDatabase = None


class RAGChain:
    """
    End-to-End RAG Chain implementing:
    1. Query embedding
    2. Vector similarity retrieval (Pinecone or FAISS)
    3. Sparse/BM25 Hybrid Reranking via RRF
    4. Grounded Prompt Formulation with context chunks
    5. LLM Answer Generation (OpenRouter or Gemini)
    """

    SYSTEM_PROMPT = """You are a strict, grounded AI assistant specialized in analyzing financial, corporate, and technical documents.

Your core instruction: Answer the user's question using ONLY the provided context chunks below.

Rules:
1. Accuracy & Grounding: Every statement and numerical figure in your response MUST be directly supported by the context chunks.
2. Numerical Precision: Quote exact figures, percentages, dates, and currency values as they appear in the source chunks. Do NOT round, estimate, or extrapolate figures unless explicitly requested.
3. Citations: Cite your sources inline using the format [Source: <Filename>, Page <PageNumber>] for every key claim or figure.
4. Missing Information: If the provided context chunks do not contain enough information to answer the question, explicitly state: "The provided document context does not contain sufficient information to answer this question." Do NOT use outside knowledge.
"""

    def __init__(
        self,
        vector_db,                          # PineconeDB or VectorDatabase (duck-typed)
        embedding_manager: EmbeddingManager,
        llm,                                # OpenRouterLLM or GeminiLLM (duck-typed)
        reranker: Optional[HybridReranker] = None,
    ):
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
        self.llm = llm
        self.reranker = reranker or HybridReranker()

    # ─── Full (non-streaming) run ─────────────────────────────────────────────

    def run(
        self,
        query: str,
        initial_top_k: int = INITIAL_TOP_K,
        rerank_top_k: int = RERANKED_TOP_K,
        filter_filenames: Optional[List[str]] = None,
        memory_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a user query.

        Args:
            query: User's question.
            initial_top_k: Number of candidates to retrieve before reranking.
            rerank_top_k: Number of chunks to keep after BM25+RRF reranking.
            filter_filenames: If set, restricts retrieval to these documents only.
            memory_context: Conversation history string to prepend to the prompt.

        Returns:
            Dict with: query, answer, retrieved_chunks, reranked_chunks,
                       citations, formatted_context.
        """
        if not query or not query.strip():
            return self._empty_response(query, "Please provide a valid question.")

        # Step 1: Embed query
        query_vec = self.embedding_manager.embed_query(query)

        # Step 2: Retrieve from vector DB (Pinecone or FAISS)
        dense_results = self._retrieve(query_vec, initial_top_k, filter_filenames)

        if not dense_results:
            return self._empty_response(
                query,
                "No indexed document chunks found. Please upload and index documents first.",
            )

        # Step 3: Hybrid BM25+RRF reranking
        reranked_results = self.reranker.rerank(
            query=query,
            dense_results=dense_results,
            top_k=rerank_top_k,
            all_chunks=self.vector_db.chunks_metadata,
        )
        top_chunks = [chunk for chunk, score in reranked_results]

        # Step 4: Build grounded prompt
        user_prompt, formatted_context = self._build_prompt(query, top_chunks, memory_context)

        # Step 5: Generate answer
        answer = self._generate(user_prompt)

        citations = self._extract_citations(top_chunks)

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": [c for c, s in dense_results],
            "reranked_chunks": top_chunks,
            "citations": citations,
            "formatted_context": formatted_context,
        }

    # ─── Streaming run ────────────────────────────────────────────────────────

    def run_stream(
        self,
        query: str,
        initial_top_k: int = INITIAL_TOP_K,
        rerank_top_k: int = RERANKED_TOP_K,
        filter_filenames: Optional[List[str]] = None,
        memory_context: Optional[str] = None,
    ) -> Generator[Tuple[str, Any], None, None]:
        """
        Streaming version of run(). Yields (event_type, data) tuples:

            ("context", reranked_chunks)   — emitted once before LLM starts
            ("token",   text_chunk)        — one per token from LLM stream
            ("done",    metadata_dict)     — final event with citations etc.

        Usage in Streamlit:
            for event_type, data in rag_chain.run_stream(query):
                if event_type == "context":
                    show_context(data)
                elif event_type == "token":
                    answer_placeholder.markdown(accumulated + data)
                elif event_type == "done":
                    store_message(data)
        """
        if not query or not query.strip():
            yield ("token", "Please provide a valid question.")
            yield ("done", {"query": query, "citations": [], "reranked_chunks": []})
            return

        # Retrieval phase (same as non-streaming)
        query_vec = self.embedding_manager.embed_query(query)
        dense_results = self._retrieve(query_vec, initial_top_k, filter_filenames)

        if not dense_results:
            yield ("token", "No indexed document chunks found. Please upload and index documents first.")
            yield ("done", {"query": query, "citations": [], "reranked_chunks": []})
            return

        reranked_results = self.reranker.rerank(
            query=query,
            dense_results=dense_results,
            top_k=rerank_top_k,
            all_chunks=self.vector_db.chunks_metadata,
        )
        top_chunks = [chunk for chunk, score in reranked_results]

        # Emit context immediately so UI can show retrieved chunks while LLM generates
        yield ("context", top_chunks)

        user_prompt, formatted_context = self._build_prompt(query, top_chunks, memory_context)
        citations = self._extract_citations(top_chunks)

        # Stream tokens from LLM
        if hasattr(self.llm, "generate_stream"):
            for token in self.llm.generate_stream(
                prompt=user_prompt,
                task="answer",
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.1,
            ):
                yield ("token", token)
        else:
            # Fallback: non-streaming generate (GeminiLLM without streaming_generate)
            answer = self.llm.generate(
                prompt=user_prompt,
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.1,
            )
            yield ("token", answer)

        yield (
            "done",
            {
                "query": query,
                "citations": citations,
                "reranked_chunks": top_chunks,
                "retrieved_chunks": [c for c, s in dense_results],
                "formatted_context": formatted_context,
            },
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _retrieve(
        self,
        query_vec,
        top_k: int,
        filter_filenames: Optional[List[str]],
    ) -> List[Tuple[Dict, float]]:
        """Routes retrieval to Pinecone or FAISS depending on the vector_db type."""
        search_kwargs: dict = {"top_k": top_k}

        # PineconeDB supports filter_filenames; VectorDatabase does not
        if filter_filenames and hasattr(self.vector_db, "search"):
            import inspect
            sig = inspect.signature(self.vector_db.search)
            if "filter_filenames" in sig.parameters:
                search_kwargs["filter_filenames"] = filter_filenames

        return self.vector_db.search(query_vec, **search_kwargs)

    def _build_prompt(
        self,
        query: str,
        top_chunks: List[Dict],
        memory_context: Optional[str],
    ) -> Tuple[str, str]:
        """Constructs the grounded user prompt with optional conversation memory."""
        context_blocks = []
        for idx, chunk in enumerate(top_chunks):
            citation_tag = f"Source: {chunk['filename']}, Page {chunk['page_number']}"
            block = f"--- CONTEXT CHUNK #{idx+1} [{citation_tag}] ---\n{chunk['text']}\n"
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)

        memory_section = ""
        if memory_context and memory_context.strip():
            memory_section = f"\nCOGNITIVE MEMORY & CONTEXT:\n{memory_context}\n"

        user_prompt = f"""CONTEXT CHUNKS:
{formatted_context}
{memory_section}
USER QUESTION:
{query}

GROUNDED ANSWER WITH INLINE CITATIONS:"""

        return user_prompt, formatted_context

    def _generate(self, user_prompt: str) -> str:
        """Calls the LLM (OpenRouter or Gemini) for a non-streaming response."""
        if not self.llm.is_available():
            return "[LLM Offline Mode — check OPENROUTER_API_KEY and GEMINI_API_KEY]"

        # OpenRouterLLM uses task="answer"; GeminiLLM uses system_instruction param
        if hasattr(self.llm, "generate") and "task" in str(type(self.llm)):
            return self.llm.generate(
                prompt=user_prompt,
                task="answer",
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.1,
            )
        else:
            return self.llm.generate(
                prompt=user_prompt,
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.1,
            )

    def _extract_citations(self, top_chunks: List[Dict]) -> List[str]:
        citations = []
        for chunk in top_chunks:
            tag = f"{chunk['filename']} (Page {chunk['page_number']})"
            if tag not in citations:
                citations.append(tag)
        return citations

    @staticmethod
    def _empty_response(query: str, message: str) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": message,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "citations": [],
            "formatted_context": "",
        }
