"""
core/retriever.py
-----------------
Two retrieval strategies:
  1. Dense (FAISS + MiniLM) — semantic similarity, handles paraphrase
  2. BM25 — keyword overlap baseline (rank_bm25 library)

Both return the same interface: List[Tuple[Chunk, float]]
This enables side-by-side evaluation in eval/evaluate.py.
"""

import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from core.ingest import Chunk
from core.embedder import FAISSIndex


class DenseRetriever:
    """Semantic retrieval via FAISS. Wraps FAISSIndex.search()."""

    def __init__(self, index: FAISSIndex):
        self.index = index

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        return self.index.search(query, top_k=top_k)


class BM25Retriever:
    """
    Keyword retrieval baseline using BM25Okapi.
    Built on the same chunk list as the dense index.
    """

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunks[idx], float(scores[idx])))

        return results


class HybridRetriever:
    """
    Optional: Reciprocal Rank Fusion of dense + BM25.
    Combine results from both retrievers for better coverage.
    Not used by default in the app — shown here as an extension.
    """

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k: int = 60):
        self.dense = dense
        self.bm25 = bm25
        self.k = k  # RRF constant

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)
        bm25_results = self.bm25.retrieve(query, top_k=top_k * 2)

        scores = {}
        for rank, (chunk, _) in enumerate(dense_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (self.k + rank + 1)
        for rank, (chunk, _) in enumerate(bm25_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (self.k + rank + 1)

        # Map chunk_id back to chunk objects
        chunk_map = {c.chunk_id: c for c, _ in dense_results + bm25_results}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [(chunk_map[cid], score) for cid, score in ranked if cid in chunk_map]
