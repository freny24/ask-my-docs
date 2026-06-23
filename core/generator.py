"""
core/generator.py — Gemini version
"""

import os
from typing import List, Tuple
from core.ingest import Chunk

SYSTEM_PROMPT = """You are a helpful research assistant that answers questions strictly based on the provided document excerpts.
- Answer only from the provided context. Do not use outside knowledge.
- If the context does not contain enough information, say so clearly.
- Always end your answer with a Sources: section listing the document name and page number.
- Keep answers concise and factual. Use bullet points for lists.
- For follow-up questions, use the conversation history to understand context."""


def _get_key():
    """Read Gemini key from Streamlit secrets or environment at call time."""
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def build_context_block(chunks):
    lines = ["### Retrieved context:\n"]
    for i, (chunk, score) in enumerate(chunks, 1):
        lines.append(
            f"[{i}] Source: {chunk.source}, Page {chunk.page} (relevance: {score:.2f})\n"
            f"{chunk.text}\n"
        )
    return "\n".join(lines)


def generate_answer(query: str, chunks, history: List[dict]):
    if not chunks:
        msg = "I couldn't find relevant information in your documents. Try rephrasing or uploading more documents."
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": msg})
        return msg, history

    import google.generativeai as genai
    genai.configure(api_key=_get_key())

    context = build_context_block(chunks)

    history_text = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n\n"

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{context}\n\n"
        f"{history_text}"
        f"User: {query}\nAssistant:"
    )

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(full_prompt)
    answer = response.text

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return answer, history


def format_sources(chunks) -> List[dict]:
    seen = set()
    sources = []
    for chunk, score in chunks:
        k = (chunk.source, chunk.page)
        if k not in seen:
            seen.add(k)
            sources.append({
                "file": chunk.source,
                "page": chunk.page,
                "score": round(score, 3),
                "preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            })
    return sources
