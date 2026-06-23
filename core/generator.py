import os
import google.generativeai as genai
from typing import List, Tuple
from core.ingest import Chunk
from dotenv import load_dotenv

load_dotenv()

# Support both .env and Streamlit secrets
try:
    import streamlit as st
    key = st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
except Exception:
    key = os.getenv("GEMINI_API_KEY", "")

genai.configure(api_key=key)

SYSTEM_PROMPT = """You are a helpful research assistant that answers questions 
strictly based on the provided document excerpts.
- Answer only from the provided context. Do not use outside knowledge.
- If the context doesn't contain enough information, say so clearly.
- Always end your answer with a Sources: section listing the document name and page number.
- Keep answers concise and factual."""


def build_context_block(chunks):
    lines = ["### Retrieved context:\n"]
    for i, (chunk, score) in enumerate(chunks, 1):
        lines.append(
            f"[{i}] Source: {chunk.source}, Page {chunk.page}\n{chunk.text}\n"
        )
    return "\n".join(lines)


def generate_answer(query, chunks, history):
    if not chunks:
        msg = "I couldn't find relevant information in your documents to answer this."
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": msg})
        return msg, history

    context = build_context_block(chunks)
    
    # Build conversation string from history
    history_text = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n\n"

    full_prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\n{history_text}User: {query}\nAssistant:"

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(full_prompt)
    answer = response.text

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return answer, history


def format_sources(chunks):
    seen = set()
    sources = []
    for chunk, score in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": chunk.source,
                "page": chunk.page,
                "score": round(score, 3),
                "preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            })
    return sources
