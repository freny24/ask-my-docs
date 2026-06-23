"""
core/generator.py
-----------------
Generates answers using Claude (claude-sonnet-4-6) with:
  - Retrieved context chunks injected into the prompt
  - Multi-turn conversation memory (last N turns)
  - Source citations in the response
"""

import os
import anthropic
from typing import List, Tuple
from core.ingest import Chunk
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a helpful research assistant that answers questions strictly based on the provided document excerpts.

Rules:
- Answer only from the provided context. Do not use outside knowledge.
- If the context doesn't contain enough information, say so clearly.
- Always end your answer with a "Sources:" section listing the document name and page number for each piece of information you used.
- Keep answers concise and factual. Use bullet points for lists.
- For follow-up questions, use the conversation history to understand what the user is referring to.
"""


def build_context_block(chunks: List[Tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = ["### Retrieved context:\n"]
    for i, (chunk, score) in enumerate(chunks, 1):
        lines.append(
            f"[{i}] Source: {chunk.source}, Page {chunk.page} (relevance: {score:.2f})\n"
            f"{chunk.text}\n"
        )
    return "\n".join(lines)


def build_messages(
    query: str,
    chunks: List[Tuple[Chunk, float]],
    history: List[dict],
    max_history_turns: int = 4,
) -> List[dict]:
    """
    Build the messages array for the Anthropic API.

    Structure:
      [recent history turns] + [context + current question]

    We inject context only in the final user turn to keep
    history turns clean and token-efficient.
    """
    messages = []

    # Include last N turns of conversation history
    recent = history[-(max_history_turns * 2):]  # 2 messages per turn
    messages.extend(recent)

    # Final turn: context + question
    context_block = build_context_block(chunks)
    user_content = f"{context_block}\n\n### Question:\n{query}"
    messages.append({"role": "user", "content": user_content})

    return messages


def generate_answer(
    query: str,
    chunks: List[Tuple[Chunk, float]],
    history: List[dict],
) -> Tuple[str, List[dict]]:
    """
    Generate an answer for the query given retrieved chunks and conversation history.

    Returns:
        answer: str — the LLM response
        updated_history: List[dict] — history with this turn appended
    """
    if not chunks:
        no_context = "I couldn't find relevant information in your documents to answer this question. Try rephrasing or uploading more documents."
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": no_context})
        return no_context, history

    messages = build_messages(query, chunks, history)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    answer = response.content[0].text

    # Update history with clean versions (no context block, just Q&A)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return answer, history


def format_sources(chunks: List[Tuple[Chunk, float]]) -> List[dict]:
    """
    Return structured source metadata for display in the UI.
    Deduplicates by (source, page).
    """
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
