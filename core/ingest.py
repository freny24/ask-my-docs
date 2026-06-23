"""
core/ingest.py
--------------
Extracts text from PDFs and splits into overlapping chunks.

Design decisions:
- PyMuPDF (fitz) for extraction: faster and more accurate than pypdf for
  multi-column layouts and scanned-adjacent PDFs.
- Recursive character splitter: respects paragraph > sentence > word
  boundaries before making a hard cut. Better than fixed-size slicing.
- 512 token target / 64 token overlap: standard for MiniLM-L6 (512 max).
  Overlap preserves context across chunk boundaries.
"""

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    text: str
    source: str        # filename
    page: int
    chunk_id: str      # "filename::page::index"


def extract_text_from_pdf(pdf_path: str) -> List[dict]:
    """
    Extract page-level text from a PDF file.
    Returns list of {text, page, source} dicts.
    """
    doc = fitz.open(pdf_path)
    pages = []
    source = Path(pdf_path).name

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:  # skip blank pages
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": source,
            })

    doc.close()
    return pages


def chunk_pages(pages: List[dict], chunk_size: int = 1800, chunk_overlap: int = 200) -> List[Chunk]:
    """
    Split page text into overlapping chunks.

    chunk_size=1800 chars ≈ 450 tokens (safe for MiniLM-L6 512 limit).
    chunk_overlap=200 chars preserves context across boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, text in enumerate(splits):
            if len(text.strip()) < 50:  # skip tiny fragments
                continue
            chunk_id = f"{page['source']}::p{page['page']}::c{i}"
            chunks.append(Chunk(
                text=text.strip(),
                source=page["source"],
                page=page["page"],
                chunk_id=chunk_id,
            ))

    return chunks


def ingest_pdfs(pdf_paths: List[str]) -> List[Chunk]:
    """
    Full ingestion: extract + chunk a list of PDF paths.
    Returns all chunks across all documents.
    """
    all_chunks = []
    for path in pdf_paths:
        try:
            pages = extract_text_from_pdf(path)
            chunks = chunk_pages(pages)
            all_chunks.extend(chunks)
            print(f"[ingest] {Path(path).name}: {len(pages)} pages → {len(chunks)} chunks")
        except Exception as e:
            print(f"[ingest] ERROR on {path}: {e}")

    print(f"[ingest] Total: {len(all_chunks)} chunks from {len(pdf_paths)} documents")
    return all_chunks
