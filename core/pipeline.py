"""
core/pipeline.py
----------------
Ties together ingest → embed → retrieve → generate.
This is what the Streamlit app imports.
"""

from typing import List, Tuple
from core.ingest import ingest_pdfs, Chunk
from core.embedder import FAISSIndex
from core.retriever import DenseRetriever, BM25Retriever
from core.generator import generate_answer, format_sources


class RAGPipeline:
    """
    Full RAG pipeline. One instance lives in Streamlit session state.

    Usage:
        pipeline = RAGPipeline()
        pipeline.load_documents(["doc1.pdf", "doc2.pdf"])
        answer, sources, history = pipeline.ask("What is X?", history=[])
    """

    def __init__(self):
        self.index = FAISSIndex()
        self.dense_retriever = None
        self.bm25_retriever = None
        self.chunks: List[Chunk] = []
        self.loaded_files: List[str] = []

    def load_documents(self, pdf_paths: List[str]):
        """Ingest PDFs, build FAISS index and BM25 index."""
        new_chunks = ingest_pdfs(pdf_paths)
        self.chunks.extend(new_chunks)
        self.loaded_files.extend(pdf_paths)

        # Rebuild indexes with all chunks
        self.index.build(self.chunks)
        self.dense_retriever = DenseRetriever(self.index)
        self.bm25_retriever = BM25Retriever(self.chunks)

    def ask(
        self,
        query: str,
        history: List[dict],
        top_k: int = 5,
    ) -> Tuple[str, List[dict], List[dict]]:
        """
        Ask a question against the loaded documents.

        Args:
            query: user's question
            history: conversation history list
            top_k: number of chunks to retrieve

        Returns:
            answer: str
            updated_history: List[dict]
            sources: List[dict] — for UI display
        """
        if self.dense_retriever is None:
            return (
                "No documents loaded yet. Please upload PDFs first.",
                history,
                [],
            )

        retrieved = self.dense_retriever.retrieve(query, top_k=top_k)
        answer, updated_history = generate_answer(query, retrieved, history)
        sources = format_sources(retrieved)

        return answer, updated_history, sources

    @property
    def is_ready(self) -> bool:
        return self.dense_retriever is not None and not self.index.is_empty

    @property
    def stats(self) -> dict:
        return {
            "documents": self.index.doc_count,
            "chunks": self.index.chunk_count,
            "files": [f.split("/")[-1] for f in self.loaded_files],
        }
