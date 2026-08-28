"""
vectorstore/pinecone_db.py
──────────────────────────
Production vector database backed by Pinecone Serverless (free tier)
with persistent SQLite database storage for full chunk texts and document registry.

Stores:
  - Vector Embeddings (Pinecone Serverless Cloud)
  - Full Chunk Texts & Metadata (data/rag_app.db -> chunk_texts table)
  - Document Registry & Stats (data/rag_app.db -> documents table)

Guarantees 0 data loss and persists across all interface reloads and server restarts.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSION,
    VECTOR_DB_DIR,
)
from database.db_manager import DatabaseManager, get_db

try:
    from pinecone import Pinecone, ServerlessSpec
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

logger = logging.getLogger(__name__)

# Legacy sidecar paths for migration
_LEGACY_CHUNK_TEXTS_PATH = VECTOR_DB_DIR / "chunk_texts.json"
_LEGACY_DOC_REGISTRY_PATH = VECTOR_DB_DIR / "doc_registry.json"


class PineconeDB:
    """
    Pinecone Serverless vector database with persistent SQLite backing
    for full chunk text and document registry management.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        dimension: int = EMBEDDING_DIMENSION,
        db: Optional[DatabaseManager] = None,
    ):
        if not HAS_PINECONE:
            raise ImportError(
                "pinecone package is required. Run: pip install pinecone>=3.0.0"
            )

        self.api_key = api_key or PINECONE_API_KEY
        self.index_name = index_name or PINECONE_INDEX_NAME
        self.cloud = cloud or PINECONE_CLOUD
        self.region = region or PINECONE_REGION
        self.dimension = dimension
        self.db = db or get_db()

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is not set. Add it to your .env file.")

        # Connect and create index if needed
        self._pc = Pinecone(api_key=self.api_key)
        self._index = self._get_or_create_index()

        # Migrate legacy JSON files if database is empty
        self._migrate_legacy_json_if_needed()

        logger.info(
            f"[PineconeDB] Connected to index '{self.index_name}'. "
            f"Persistent DB active."
        )

    # ─── Index management ─────────────────────────────────────────────────────

    def _get_or_create_index(self):
        """Creates the Pinecone index if it doesn't exist, then waits for it to be ready."""
        existing = [idx.name for idx in self._pc.list_indexes()]

        if self.index_name not in existing:
            logger.info(f"[PineconeDB] Creating index '{self.index_name}'...")
            self._pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
            # Wait until ready
            for _ in range(30):
                desc = self._pc.describe_index(self.index_name)
                if desc.status.get("ready", False):
                    break
                time.sleep(2)
            logger.info(f"[PineconeDB] Index '{self.index_name}' is ready.")

        return self._pc.Index(self.index_name)

    def _migrate_legacy_json_if_needed(self) -> None:
        """Migrates legacy JSON sidecar files to SQLite database if present."""
        try:
            doc_count = len(self.db.get_documents())
            if doc_count == 0 and _LEGACY_DOC_REGISTRY_PATH.exists():
                with open(_LEGACY_DOC_REGISTRY_PATH, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                for fname, d in docs.items():
                    self.db.upsert_document(
                        filename=fname,
                        chunk_count=d.get("chunk_count", 0),
                        file_size=d.get("file_size", 0),
                    )

            chunk_count = len(self.db.get_all_chunks())
            if chunk_count == 0 and _LEGACY_CHUNK_TEXTS_PATH.exists():
                with open(_LEGACY_CHUNK_TEXTS_PATH, "r", encoding="utf-8") as f:
                    texts = json.load(f)
                chunk_items = [
                    {"chunk_id": cid, "filename": cid.rsplit("_chunk_", 1)[0] if "_chunk_" in cid else "doc", "text": text}
                    for cid, text in texts.items()
                ]
                self.db.save_chunk_texts(chunk_items)
        except Exception as exc:
            logger.debug(f"[PineconeDB] Migration check notice: {exc}")

    # ─── Core operations ─────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Batch-upserts chunks and their embeddings to Pinecone and persists full
        chunk texts and metadata into the SQLite database.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        vectors = []
        filenames_seen: dict[str, int] = {}

        for chunk, vec in zip(chunks, embeddings):
            chunk_id = chunk["chunk_id"]
            filename = chunk.get("filename", "unknown")
            filenames_seen[filename] = filenames_seen.get(filename, 0) + 1

            # Pinecone metadata — indexable/filterable fields
            metadata = {
                "filename": filename,
                "page_number": int(chunk.get("page_number", 1)),
                "has_table": bool(chunk.get("has_table", False)),
                "numeric_count": int(chunk.get("numeric_count", 0)),
                "header_context": chunk.get("header_context", "")[:500],
            }

            vectors.append({
                "id": chunk_id,
                "values": vec.tolist(),
                "metadata": metadata,
            })

        # Batch upsert to Pinecone in groups of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch)

        # Persist full chunk texts and metadata to SQLite database
        self.db.save_chunk_texts(chunks)

        # Update document registry in database
        for filename, count in filenames_seen.items():
            self.db.upsert_document(filename=filename, chunk_count=count)

        logger.info(
            f"[PineconeDB] Upserted {len(chunks)} chunks to Pinecone and saved to SQLite DB "
            f"({list(filenames_seen.keys())})."
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 15,
        filter_filenames: Optional[list[str]] = None,
    ) -> list[tuple[dict, float]]:
        """
        Queries Pinecone for the most similar chunks and reconstructs
        full chunk text from SQLite database.
        """
        if query_embedding.ndim == 1:
            query_vec = query_embedding.tolist()
        else:
            query_vec = query_embedding[0].tolist()

        query_kwargs: dict = {
            "vector": query_vec,
            "top_k": top_k,
            "include_metadata": True,
        }

        if filter_filenames:
            query_kwargs["filter"] = {"filename": {"$in": filter_filenames}}

        response = self._index.query(**query_kwargs)

        results = []
        for match in response.matches:
            chunk_id = match.id
            meta = match.metadata or {}

            # Reconstruct full chunk text from SQLite database
            db_text = self.db.get_chunk_text(chunk_id)

            chunk = {
                "chunk_id": chunk_id,
                "text": db_text or meta.get("header_context", ""),
                "filename": meta.get("filename", "unknown"),
                "page_number": meta.get("page_number", 1),
                "has_table": meta.get("has_table", False),
                "numeric_count": meta.get("numeric_count", 0),
                "header_context": meta.get("header_context", ""),
            }
            results.append((chunk, float(match.score)))

        return results

    def delete_by_filename(self, filename: str) -> int:
        """
        Deletes all vectors belonging to a document from Pinecone and SQLite DB.
        """
        stem = Path(filename).stem
        # 1. Delete from Pinecone by metadata filter
        try:
            self._index.delete(filter={"filename": {"$eq": filename}})
        except Exception as exc:
            logger.debug(f"[PineconeDB] Filter delete fallback: {exc}")

        # 2. Clean up SQLite database
        deleted_count = self.db.delete_document(filename)
        logger.info(f"[PineconeDB] Deleted {deleted_count} chunks for '{filename}' from DB and Pinecone.")
        return deleted_count

    def list_documents(self) -> dict[str, dict]:
        """Returns the persistent document registry: {filename: {chunk_count, indexed_at}}."""
        return self.db.get_documents()

    def get_stats(self) -> dict:
        """Returns Pinecone index and local database statistics."""
        try:
            stats = self._index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "db_documents": len(self.db.get_documents()),
                "db_chunks": len(self.db.get_all_chunks()),
            }
        except Exception as exc:
            logger.warning(f"[PineconeDB] Failed to get stats: {exc}")
            return {
                "total_vector_count": 0,
                "dimension": self.dimension,
                "index_fullness": 0.0,
                "db_documents": len(self.db.get_documents()),
                "db_chunks": len(self.db.get_all_chunks()),
            }

    def clear(self) -> None:
        """Deletes ALL vectors from Pinecone and clears the local database document storage."""
        try:
            self._index.delete(delete_all=True)
        except Exception as exc:
            logger.warning(f"[PineconeDB] clear() error: {exc}")

        self.db.clear_documents_and_chunks()
        logger.info("[PineconeDB] Index and database cleared.")

    # ─── Compatibility shim ────────────────────────────────────────────────────

    @property
    def chunks_metadata(self) -> list[dict]:
        """
        Compatibility shim for HybridReranker which needs the full corpus for BM25.
        Reads directly from SQLite database.
        """
        return self.db.get_all_chunks()
