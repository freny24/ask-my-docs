"""
core/embedder.py
----------------
Embeds chunks using sentence-transformers and stores in a FAISS index.

Model: all-MiniLM-L6-v2
- 384-dim embeddings, ~22M params
- Runs locally, no API cost
- Fast: ~14k sentences/sec on CPU
- Hits 90%+ of OpenAI Ada-002 on MTEB benchmarks
"""

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from core.ingest import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # lazy load


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


class FAISSIndex:
    """
    Wraps a FAISS flat index with chunk metadata.
    Supports save/load to disk for session persistence.
    """

    def __init__(self):
        self.index = None
        self.chunks: List[Chunk] = []
        self.dim = 384  # MiniLM-L6 output dim

    def build(self, chunks: List[Chunk], batch_size: int = 64):
        """Embed all chunks and build the FAISS index."""
        model = get_model()
        texts = [c.text for c in chunks]

        print(f"[embedder] Embedding {len(texts)} chunks...")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # enables cosine sim via inner product
        )
        embeddings = np.array(embeddings, dtype="float32")

        # IndexFlatIP = inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.chunks = chunks
        print(f"[embedder] FAISS index built: {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks for a query.
        Returns [(chunk, score), ...] sorted by relevance.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        model = get_model()
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))

        return results

    def save(self, path: str):
        """Save index + metadata to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        with open(p / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[embedder] Saved index to {path}")

    def load(self, path: str) -> bool:
        """Load index from disk. Returns True if successful."""
        p = Path(path)
        faiss_path = p / "index.faiss"
        chunks_path = p / "chunks.pkl"
        if not faiss_path.exists() or not chunks_path.exists():
            return False
        self.index = faiss.read_index(str(faiss_path))
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"[embedder] Loaded index: {self.index.ntotal} vectors")
        return True

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    @property
    def doc_count(self) -> int:
        return len(set(c.source for c in self.chunks))

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
