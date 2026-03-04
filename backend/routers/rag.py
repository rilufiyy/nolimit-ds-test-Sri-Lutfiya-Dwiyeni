"""
RAG chatbot router.

Endpoints:
  POST /api/rag/chat/{meeting_id}    - Q&A grounded in transcript
  POST /api/rag/reindex/{meeting_id} - force rebuild FAISS index
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from models.schemas import RAGChatRequest, RAGChatResponse
from services.rag_chain import rag_chain
from services.storage import storage_service
from services.vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["RAG Chatbot"])


def _require_completed(meeting_id: str) -> None:
    """Raise 404/409 if the meeting is missing or not yet completed."""
    info = storage_service.get_meeting_info(meeting_id)
    if not info:
        raise HTTPException(404, "Meeting not found.")
    if info.status != "completed":
        raise HTTPException(409, f"Meeting not ready. Current status: {info.status}")


@router.post("/chat/{meeting_id}", response_model=RAGChatResponse)
async def chat_with_meeting(meeting_id: str, request: RAGChatRequest):
    """
    Ask a question about a specific meeting.

    The answer is grounded in retrieved transcript chunks (RAG).
    Each response includes **source citations** (speaker, timestamp, text).

    - **top_k**: number of transcript chunks to retrieve (1-20, default 5).
    """
    _require_completed(meeting_id)

    if not vector_store.exists(meeting_id):
        result = storage_service.load_meeting(meeting_id)
        if not result:
            raise HTTPException(500, "Transcript data not found.")
        texts = [s.text for s in result.segments]
        meta = [
            {"speaker": s.speaker, "start_time": s.start_time, "end_time": s.end_time}
            for s in result.segments
        ]
        vector_store.index_meeting(meeting_id, texts, meta)
        logger.info(f"Lazily rebuilt FAISS index for meeting {meeting_id}.")

    history = [t.model_dump() for t in request.history] if request.history else None
    try:
        return rag_chain.chat(meeting_id, request.query, request.top_k, history)
    except Exception as exc:
        logger.error(f"RAG chat error for {meeting_id}: {exc}")
        raise HTTPException(500, f"RAG error: {exc}")


@router.post("/reindex/{meeting_id}", status_code=200)
async def reindex_meeting(meeting_id: str):
    """Force rebuild the FAISS index for a meeting (useful after edits)."""
    _require_completed(meeting_id)
    result = storage_service.load_meeting(meeting_id)
    if not result:
        raise HTTPException(404, "Transcript data not found.")

    texts = [s.text for s in result.segments]
    meta = [
        {"speaker": s.speaker, "start_time": s.start_time, "end_time": s.end_time}
        for s in result.segments
    ]
    vector_store.index_meeting(meeting_id, texts, meta)
    return {"message": f"Re-indexed {len(texts)} chunks for meeting {meeting_id}."}