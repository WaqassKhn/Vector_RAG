import re
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from config import RRF_K, RERANKED_TOP_K

class HybridReranker:
    """
    Reranks dense vector search candidates using BM25 keyword matching and Reciprocal Rank Fusion (RRF).
    Boosts chunks containing exact numerical tokens present in the user query.
    """

    def __init__(self, rrf_k: int = RRF_K):
        self.rrf_k = rrf_k

    def _tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer preserving numbers and symbols."""
        return re.findall(r'\b\w+(?:\.\w+)?\b', text.lower())

    def rerank(
        self,
        query: str,
        dense_results: List[Tuple[Dict[str, Any], float]],
        top_k: int = RERANKED_TOP_K,
        all_chunks: List[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Merges dense vector retrieval with BM25 sparse retrieval via RRF score.
        """
        if not dense_results:
            return []

        # If all_chunks is not provided, use dense candidate set as corpus
        corpus_chunks = all_chunks if all_chunks else [doc for doc, _ in dense_results]
        
        # Tokenize corpus for BM25
        tokenized_corpus = [self._tokenize(doc["text"]) for doc in corpus_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        query_tokens = self._tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)

        # Map chunk_id to BM25 rank
        bm25_ranked_indices = list(np.argsort(bm25_scores)[::-1]) if len(bm25_scores) > 0 else []
        bm25_rank_map = {corpus_chunks[idx]["chunk_id"]: rank + 1 for rank, idx in enumerate(bm25_ranked_indices)}

        # Dense rank map
        dense_rank_map = {doc["chunk_id"]: rank + 1 for rank, (doc, _) in enumerate(dense_results)}

        # Extract numeric tokens from query
        query_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', query))

        # Calculate RRF score for all candidate chunks in dense_results
        rrf_scored_chunks = []
        chunk_map = {doc["chunk_id"]: doc for doc, _ in dense_results}

        for chunk_id, doc in chunk_map.items():
            r_dense = dense_rank_map.get(chunk_id, 999)
            r_sparse = bm25_rank_map.get(chunk_id, 999)

            score_dense = 1.0 / (self.rrf_k + r_dense)
            score_sparse = 1.0 / (self.rrf_k + r_sparse)
            
            total_rrf = score_dense + score_sparse

            # Numerical token bonus
            if query_numbers:
                chunk_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', doc["text"]))
                matched_nums = query_numbers.intersection(chunk_numbers)
                if matched_nums:
                    # Give boost proportional to matched numbers
                    total_rrf += 0.05 * len(matched_nums)

            rrf_scored_chunks.append((doc, total_rrf))

        # Sort descending by final RRF score
        rrf_scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return rrf_scored_chunks[:top_k]

import numpy as np
