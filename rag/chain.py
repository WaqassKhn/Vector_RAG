import re
from typing import List, Dict, Any, Tuple, Optional
from vectorstore.embeddings import EmbeddingManager
from vectorstore.vector_db import VectorDatabase
from rag.reranker import HybridReranker
from rag.llm import GeminiLLM
from config import INITIAL_TOP_K, RERANKED_TOP_K

class RAGChain:
    """
    End-to-End RAG Chain implementing:
    1. Query embedding
    2. Vector similarity retrieval
    3. Sparse/BM25 Hybrid Reranking
    4. Grounded Prompt Formulation
    5. Gemini LLM Answer Generation with explicit citations
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
        vector_db: VectorDatabase,
        embedding_manager: EmbeddingManager,
        llm: GeminiLLM,
        reranker: Optional[HybridReranker] = None
    ):
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
        self.llm = llm
        self.reranker = reranker or HybridReranker()

    def run(
        self,
        query: str,
        initial_top_k: int = INITIAL_TOP_K,
        rerank_top_k: int = RERANKED_TOP_K
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a user query.
        """
        if not query or not query.strip():
            return {
                "query": query,
                "answer": "Please provide a valid question.",
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "citations": []
            }

        # Step 1: Question Embedding
        query_vec = self.embedding_manager.embed_query(query)

        # Step 2: Semantic Similarity Search in Vector DB
        dense_results = self.vector_db.search(query_vec, top_k=initial_top_k)

        if not dense_results:
            return {
                "query": query,
                "answer": "No indexed document chunks found in vector database. Please index documents first.",
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "citations": []
            }

        # Step 3: Hybrid Reranking & Filtering
        reranked_results = self.reranker.rerank(
            query=query,
            dense_results=dense_results,
            top_k=rerank_top_k,
            all_chunks=self.vector_db.chunks_metadata
        )

        top_chunks = [chunk for chunk, score in reranked_results]

        # Step 4: Construct Grounded Prompt with Context Chunks
        context_blocks = []
        for idx, chunk in enumerate(top_chunks):
            citation_tag = f"Source: {chunk['filename']}, Page {chunk['page_number']}"
            block = f"--- CONTEXT CHUNK #{idx+1} [{citation_tag}] ---\n{chunk['text']}\n"
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)
        
        user_prompt = f"""CONTEXT CHUNKS:
{formatted_context}

USER QUESTION:
{query}

GROUNDED ANSWER WITH INLINE CITATIONS:"""

        # Step 5: LLM Generation
        if not self.llm.is_available():
            # Fallback if Gemini key is missing
            answer = f"[LLM Offline Mode - Gemini API Key missing]\n\nTop Retrieved Context:\n{formatted_context}"
        else:
            answer = self.llm.generate(
                prompt=user_prompt,
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.1
            )

        # Extract cited sources
        citations = []
        for chunk in top_chunks:
            tag = f"{chunk['filename']} (Page {chunk['page_number']})"
            if tag not in citations:
                citations.append(tag)

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": [c for c, s in dense_results],
            "reranked_chunks": top_chunks,
            "citations": citations,
            "formatted_context": formatted_context
        }
