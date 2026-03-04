"""
Sentiment analysis router.

Endpoint:
  POST /api/sentiment/{meeting_id}  - run per-speaker sentiment analysis
  GET  /api/sentiment/{meeting_id}  - retrieve cached sentiment result
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from models.schemas import SentimentResult
from services.sentiment import sentiment_service
from services.storage import storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sentiment", tags=["Sentiment Analysis"])


def _require_completed(meeting_id: str) -> None:
    info = storage_service.get_meeting_info(meeting_id)
    if not info:
        raise HTTPException(404, "Meeting not found.")
    if info.status != "completed":
        raise HTTPException(409, f"Meeting not ready. Status: {info.status}")


@router.post("/{meeting_id}", response_model=SentimentResult)
async def analyse_sentiment(meeting_id: str):
    """
    Run per-speaker sentiment analysis on a completed meeting transcript.

    Uses cardiffnlp/twitter-roberta-base-sentiment (local, no API).
    Results are computed on-demand and returned as JSON.

    Returns:
        SentimentResult with speaker_sentiment mapping:
        {speaker: {positive, neutral, negative}} (all values sum ≈ 1.0)
    """
    _require_completed(meeting_id)

    result = storage_service.load_meeting(meeting_id)
    if not result:
        raise HTTPException(404, "Transcript data not found.")
    if not result.segments:
        raise HTTPException(400, "No transcript segments to analyse.")

    loop = asyncio.get_running_loop()
    try:
        sentiment = await loop.run_in_executor(
            None,
            lambda: sentiment_service.analyse(result.segments),
        )
    except Exception as exc:
        logger.error(f"Sentiment analysis failed for {meeting_id}: {exc}", exc_info=True)
        raise HTTPException(500, f"Sentiment analysis error: {exc}")
    sentiment.meeting_id = meeting_id
    return sentiment