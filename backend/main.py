"""
MeetRecall AI - FastAPI backend entry point.

Services:
  - Transcription  : Whisper-small (local) + AssemblyAI (cloud)
  - Embeddings     : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - Vector Search  : FAISS (per-meeting index, persisted to disk)
  - RAG Chatbot    : google/flan-t5-base grounded on transcript chunks
  - Topic Cluster  : KMeans / HDBSCAN on segment embeddings
"""
from __future__ import annotations

import logging
import shutil

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import clustering, rag, sentiment, transcribe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="MeetRecall AI",
    version="2.0.0",
    description=(
        "Meeting intelligence platform - transcription, RAG chatbot, and topic clustering. "
        "All NLP models are served from HuggingFace."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe.router)
app.include_router(rag.router)
app.include_router(clustering.router)
app.include_router(sentiment.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "app": "MeetRecall AI",
        "version": "2.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Quick liveness check - also verifies ffmpeg is installed."""
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    return {
        "status": "healthy" if ffmpeg_ok else "degraded",
        "ffmpeg": ffmpeg_ok,
    }