"""
RAG Vector Store — Part B Implementation
==========================================
Manual FAISS index wrapping.
Provides:
  - IndexFlatIP (inner product = cosine on L2-normalised embeddings)
  - Top-k retrieval with similarity scores
  - Metadata-aware result objects
  - Save / load FAISS index from disk
"""

import os
import pickle
import numpy as np
import faiss
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

from .chunker import Chunk


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float            # cosine similarity in [0, 1] (higher = better)
    rank: int               # 1-based rank


class FAISSVectorStore:
    """
    Wraps a FAISS flat inner-product index.

    Design decisions:
      • IndexFlatIP: exact (non-approximate) search — acceptable for corpus
        sizes up to ~100k chunks, which covers both our datasets.
      • L2-normalised embeddings → IP = cosine similarity.
      • Chunk metadata stored in a parallel Python list (no SQL dependency).
    """

    INDEX_PATH  = "data/faiss.index"
    CHUNKS_PATH = "data/chunks_store.pkl"

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Chunk] = []
        print(f"[VectorStore] Initialised FAISS IndexFlatIP (dim={embedding_dim})")

    # ──────────────────────────────────────────────────────────────────
    #  INDEX BUILDING
    # ──────────────────────────────────────────────────────────────────

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        """
        Add chunks + their precomputed embeddings to the index.
        embeddings: shape (N, embedding_dim), float32, L2-normalised.
        """
        assert embeddings.shape[0] == len(chunks), "Chunk/embedding count mismatch"
        assert embeddings.shape[1] == self.embedding_dim, "Dimension mismatch"

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"[VectorStore] Added {len(chunks)} vectors. "
              f"Total indexed: {self.index.ntotal}")

    # ──────────────────────────────────────────────────────────────────
    #  TOP-K RETRIEVAL
    # ──────────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve the top-k most similar chunks.

        Args:
            query_embedding: shape (1, embedding_dim), L2-normalised.
            k: number of results.

        Returns:
            List of RetrievalResult sorted by descending similarity score.
        """
        if self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty. Call add() first.")

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results: List[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:     # FAISS returns -1 for unfilled slots
                continue
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(score),
                rank=rank,
            ))

        return results

    # ──────────────────────────────────────────────────────────────────
    #  FILTERED SEARCH  (source-aware)
    # ──────────────────────────────────────────────────────────────────

    def search_filtered(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        source_filter: str = None,   # 'csv' | 'pdf' | None
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k results optionally filtered by source type.
        We over-fetch (3× k) then apply the filter post-hoc.
        """
        raw = self.search(query_embedding, k=min(k * 3, self.index.ntotal))
        if source_filter:
            raw = [r for r in raw if r.chunk.source == source_filter]
        return raw[:k]

    # ──────────────────────────────────────────────────────────────────
    #  PERSISTENCE
    # ──────────────────────────────────────────────────────────────────

    def save(self, index_path: str = INDEX_PATH, chunks_path: str = CHUNKS_PATH):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[VectorStore] Saved FAISS index to {index_path}")

    def load(self, index_path: str = INDEX_PATH, chunks_path: str = CHUNKS_PATH) -> bool:
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            return False
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"[VectorStore] Loaded {self.index.ntotal} vectors from {index_path}")
        return True

    # ──────────────────────────────────────────────────────────────────
    #  STATS
    # ──────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        csv_count = sum(1 for c in self.chunks if c.source == "csv")
        pdf_count = sum(1 for c in self.chunks if c.source == "pdf")
        return {
            "total_vectors": self.index.ntotal,
            "csv_chunks":    csv_count,
            "pdf_chunks":    pdf_count,
            "dimension":     self.embedding_dim,
        }
