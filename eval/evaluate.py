"""
eval/evaluate.py
----------------
Evaluates retrieval quality on a labeled QA dataset.

Metrics:
  Hit Rate @k  — fraction of queries where the correct chunk appears in top-k
  MRR @k       — Mean Reciprocal Rank: rewards finding the right chunk higher up

Run:
    python -m eval.evaluate --pdf_dir data/sample_docs --eval_file eval/eval_dataset.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from core.ingest import ingest_pdfs
from core.embedder import FAISSIndex
from core.retriever import DenseRetriever, BM25Retriever


def hit_rate(results, relevant_source: str, relevant_page: int) -> bool:
    """Returns True if the correct source+page appears in retrieved results."""
    for chunk, _ in results:
        if chunk.source == relevant_source and chunk.page == relevant_page:
            return True
    return False


def reciprocal_rank(results, relevant_source: str, relevant_page: int) -> float:
    """Returns 1/rank of the first correct result, or 0 if not found."""
    for rank, (chunk, _) in enumerate(results, 1):
        if chunk.source == relevant_source and chunk.page == relevant_page:
            return 1.0 / rank
    return 0.0


def evaluate(pdf_dir: str, eval_file: str, top_k: int = 5):
    """
    Run evaluation comparing Dense vs BM25 retrieval.

    eval_dataset.json format:
    [
      {
        "question": "What is...",
        "answer": "...",
        "source": "document.pdf",
        "page": 3
      },
      ...
    ]
    """
    # Load eval dataset
    with open(eval_file) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} eval questions")

    # Ingest documents
    pdf_paths = list(Path(pdf_dir).glob("*.pdf"))
    chunks = ingest_pdfs([str(p) for p in pdf_paths])

    # Build retrievers
    index = FAISSIndex()
    index.build(chunks)
    dense = DenseRetriever(index)
    bm25 = BM25Retriever(chunks)

    dense_hits, dense_rrs = [], []
    bm25_hits, bm25_rrs = [], []

    for item in dataset:
        q = item["question"]
        src = item["source"]
        page = item["page"]

        dense_results = dense.retrieve(q, top_k=top_k)
        bm25_results = bm25.retrieve(q, top_k=top_k)

        dense_hits.append(hit_rate(dense_results, src, page))
        dense_rrs.append(reciprocal_rank(dense_results, src, page))

        bm25_hits.append(hit_rate(bm25_results, src, page))
        bm25_rrs.append(reciprocal_rank(bm25_results, src, page))

    print(f"\n{'Metric':<20} {'Dense (FAISS)':<20} {'BM25 Baseline':<20} {'Delta'}")
    print("-" * 70)

    dense_hr = np.mean(dense_hits)
    bm25_hr = np.mean(bm25_hits)
    print(f"{'Hit Rate @'+str(top_k):<20} {dense_hr:.3f}{'':>14} {bm25_hr:.3f}{'':>14} {dense_hr - bm25_hr:+.3f}")

    dense_mrr = np.mean(dense_rrs)
    bm25_mrr = np.mean(bm25_rrs)
    print(f"{'MRR @'+str(top_k):<20} {dense_mrr:.3f}{'':>14} {bm25_mrr:.3f}{'':>14} {dense_mrr - bm25_mrr:+.3f}")

    print(f"\nEval set size: {len(dataset)} questions | Top-k: {top_k}")

    return {
        "dense_hit_rate": dense_hr,
        "dense_mrr": dense_mrr,
        "bm25_hit_rate": bm25_hr,
        "bm25_mrr": bm25_mrr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="data/sample_docs")
    parser.add_argument("--eval_file", default="eval/eval_dataset.json")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.pdf_dir, args.eval_file, args.top_k)
