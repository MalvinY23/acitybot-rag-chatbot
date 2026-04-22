"""
RAG Embedder — Part B Implementation
======================================
Custom embedding pipeline using sentence-transformers (all-MiniLM-L6-v2).
No LangChain / LlamaIndex wrappers — all logic is implemented here.

Model choice justification:
  • all-MiniLM-L6-v2: 22M params, 384-dim embeddings, ~60 ms/query on CPU.
    Excellent trade-off between quality and speed for a capstone demo.
  • Produces normalised L2 embeddings → cosine similarity = dot product.
  • Handles up to 256 word-piece tokens per passage (our 512-char chunks
    average ~100 tokens, so we stay well within limits).
"""

import os
import pickle
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .chunker import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_PATH = "data/embeddings_cache.pkl"


class EmbeddingPipeline:
    """
    Wraps a SentenceTransformer model and provides:
      - batch encoding for a list of Chunk objects
      - single-query encoding
      - persistence (save/load from disk)
    """

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[Embedder] Loading model: {model_name} …")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Embedding dimension: {self.embedding_dim}")

    # ──────────────────────────────────────────────────────────────────
    #  ENCODE CHUNKS  (batch, with progress bar)
    # ──────────────────────────────────────────────────────────────────

    def encode_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of Chunk objects.
        Returns a float32 numpy array of shape (N, embedding_dim).

        Steps:
          1. Extract text from each Chunk.
          2. Prepend a passage prefix for asymmetric search performance.
          3. Batch-encode using the SentenceTransformer.
          4. Normalise to unit-L2 so cosine similarity = dot product.
        """
        texts = [f"passage: {c.text}" for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2-normalise
            convert_to_numpy=True,
        )
        print(f"[Embedder] Encoded {len(chunks)} chunks → shape {embeddings.shape}")
        return embeddings.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────
    #  ENCODE QUERY  (single string, may be expanded)
    # ──────────────────────────────────────────────────────────────────

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.
        Uses a query prefix (asymmetric search: query vs passage).
        Returns shape (1, embedding_dim) float32.
        """
        prefixed = f"query: {query}"
        emb = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────
    #  PERSISTENCE
    # ──────────────────────────────────────────────────────────────────

    def save(self, chunks: List[Chunk], embeddings: np.ndarray, path: str = CACHE_PATH):
        """Pickle (chunks, embeddings) to disk to avoid re-computing."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": chunks, "embeddings": embeddings,
                         "model": self.model_name}, f)
        print(f"[Embedder] Saved {len(chunks)} embeddings to {path}")

    @staticmethod
    def load(path: str = CACHE_PATH):
        """
        Load cached (chunks, embeddings) from disk.
        Returns (chunks, embeddings) or (None, None) if not found.
        """
        if not os.path.exists(path):
            return None, None
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"[Embedder] Loaded {len(data['chunks'])} cached embeddings from {path}")
        return data["chunks"], data["embeddings"]
