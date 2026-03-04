"""
RAG (Retrieval-Augmented Generation) chain — meeting intelligence edition.

Pipeline:
  1. Embed the user query with sentence-transformers.
  2. Retrieve top-k (default 3) most relevant chunks from FAISS.
  3. Build a compact context string (≤ 1200 chars).
  4. Generate a concise, factual answer via the shared LLMService.
  5. Post-process: extract key_actions and keywords from retrieved context.
  6. Return RAGChatResponse (pure JSON, no HTML).

Performance targets (Section F):
  top_k         = 3 default
  context_chars ≤ 1200
  max_new_tokens = 150
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

from models.schemas import RAGChatResponse, SourceChunk
from services.llm import llm_service
from services.vector_store import vector_store

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS: int = 2400

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "we", "our", "you", "your", "they", "their",
    "he", "she", "his", "her", "i", "my", "me", "us", "in", "on", "at",
    "to", "for", "of", "and", "or", "but", "with", "from", "by", "as",
    "if", "so", "not", "no", "yes", "also", "just", "there", "here",
    "when", "where", "what", "how", "who", "which", "than", "then",
    "about", "up", "out", "get", "got", "let", "say", "said", "know",
    "think", "make", "go", "going", "one", "two", "all", "any", "some",
    "like", "more", "very", "really", "okay", "um", "uh", "yeah",
    "right", "well", "now", "back", "see",
}

_ACTION_KEYWORDS = {
    "will", "should", "must", "need to", "going to", "plan to",
    "decided", "action item", "follow up", "todo", "next step",
    "responsible", "assign", "deadline", "deliver", "commit",
}


def _build_context(chunks: List[Dict]) -> str:
    lines, total = [], 0
    for c in chunks:
        line = (
            f"[{c.get('speaker', 'Unknown')} @ {c.get('start_time', 0):.1f}s]: "
            f"{c['text']}"
        )
        if total + len(line) > MAX_CONTEXT_CHARS:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


_MAX_HISTORY_TURNS = 3   


def _build_prompt(query: str, context: str, history: list | None = None) -> str:
    """
    Build a conversational RAG prompt.

    Includes prior conversation turns (up to _MAX_HISTORY_TURNS) so the LLM
    can handle follow-up questions like "tell me more" or "who said that?".
    """
    history_block = ""
    if history:
        turns = history[-_MAX_HISTORY_TURNS:]
        lines = ["Previous conversation:"]
        for t in turns:
            lines.append(f"User: {t.get('question', t.get('q', ''))}")
            lines.append(f"Assistant: {t.get('answer', t.get('a', ''))}")
        history_block = "\n".join(lines) + "\n\n"

    return (
        "You are a helpful AI assistant that analyzes meeting transcripts. "
        "Answer the user's question based on the meeting transcript provided below. "
        "Write a clear, complete, and natural answer in full sentences. "
        "If the user asks about action items or decisions, list them explicitly. "
        "If the transcript does not contain enough information to answer, say so honestly.\n\n"
        f"{history_block}"
        f"Meeting Transcript:\n{context}\n\n"
        f"User: {query}\n"
        "Assistant:"
    )


def _extract_keywords(context: str, top_n: int = 8) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", context.lower())
    freq: Dict[str, int] = {}
    for tok in tokens:
        if tok not in _STOPWORDS:
            freq[tok] = freq.get(tok, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]


def _extract_key_actions(context: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", context)
    actions: List[str] = []
    for sent in sentences:
        lower = sent.lower()
        if any(kw in lower for kw in _ACTION_KEYWORDS):
            cleaned = sent.strip()
            if len(cleaned) > 15:
                actions.append(cleaned)
    return actions[:5]


class RAGChain:
    """Retrieval-Augmented Generation over a single meeting's transcript."""

    def chat(
        self, meeting_id: str, query: str, top_k: int = 3,
        history: list | None = None,
    ) -> RAGChatResponse:
        """
        Retrieve relevant chunks and generate a structured grounded answer.

        Returns RAGChatResponse with transcription_summary, key_actions,
        keywords, and source citations.
        """
        chunks = vector_store.search(meeting_id, query, top_k)

        if not chunks:
            return RAGChatResponse(
                meeting_id=meeting_id,
                transcription_summary=(
                    "No relevant information found in this meeting's transcript. "
                    "Ensure transcription has completed and the FAISS index is built."
                ),
                key_actions=[],
                keywords=[],
                sources=[],
            )

        context = _build_context(chunks)
        summary = llm_service.generate(
            _build_prompt(query, context, history), max_new_tokens=300, num_beams=1
        )

        return RAGChatResponse(
            meeting_id=meeting_id,
            transcription_summary=summary,
            key_actions=_extract_key_actions(context),
            keywords=_extract_keywords(context),
            sources=[
                SourceChunk(
                    text=c["text"],
                    speaker=c.get("speaker", "Unknown"),
                    start_time=c.get("start_time", 0.0),
                    end_time=c.get("end_time", 0.0),
                    score=c.get("score", 0.0),
                )
                for c in chunks
            ],
        )


rag_chain = RAGChain()