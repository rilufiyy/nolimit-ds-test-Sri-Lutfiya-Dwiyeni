"""
Transcription router.

Endpoints:
  POST   /api/transcribe              - upload file, start async pipeline
  GET    /api/status/{meeting_id}     - lightweight status poll
  GET    /api/transcript/{meeting_id} - full transcript (only when completed)
  GET    /api/meetings                - list all meetings (newest first)
  DELETE /api/transcript/{meeting_id} - remove meeting and its data
"""
from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from models.schemas import MeetingInfo, TranscriptionResult
from services.storage import storage_service
from services.transcription import transcription_service
from services.vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Transcription"])

_UPLOAD_DIR = Path("uploads")
_UPLOAD_DIR.mkdir(exist_ok=True)

_ALLOWED_EXT = {".mp4", ".mkv", ".mov", ".avi", ".mp3", ".wav", ".m4a", ".webm"}


# Background pipeline

async def _pipeline(audio_path: Path, meeting_id: str, filename: str, provider: str) -> None:
    """Transcribe -> embed -> index. Runs in FastAPI background task."""
    storage_service.update_status(meeting_id, "processing")
    try:
        segments, full_text, duration = await transcription_service.transcribe(
            audio_path, provider=provider
        )
        result = TranscriptionResult(
            meeting_id=meeting_id,
            filename=filename,
            duration=duration,
            segments=segments,
            full_text=full_text,
            provider=provider,
            created_at=datetime.utcnow(),
            status="completed",
        )
        storage_service.save_meeting(result)

        # Build FAISS index so RAG and clustering are immediately available
        texts = [s.text for s in segments]
        meta = [
            {"speaker": s.speaker, "start_time": s.start_time, "end_time": s.end_time}
            for s in segments
        ]
        vector_store.index_meeting(meeting_id, texts, meta)
        logger.info(f"Pipeline completed for meeting {meeting_id}.")
    except Exception as exc:
        logger.error(f"Pipeline failed for {meeting_id}: {exc}")
        storage_service.update_status(meeting_id, "failed")
    finally:
        audio_path.unlink(missing_ok=True)


# Endpoints

@router.post("/transcribe", status_code=202)
async def upload_and_transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    provider: str = Form("local"),  
):
    """
    Upload an audio/video file and start transcription in the background.

    - **provider**: `local` (Whisper + Pyannote, free) or `assemblyai` (cloud, fast).
    - Returns a `meeting_id` immediately; poll `/api/status/{meeting_id}` for progress.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(_ALLOWED_EXT)}")

    meeting_id = str(uuid.uuid4())
    save_path = _UPLOAD_DIR / f"{meeting_id}{ext}"

    try:
        with save_path.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except Exception as exc:
        raise HTTPException(500, f"File save failed: {exc}")

    # Register placeholder so status endpoint works immediately
    storage_service.create_meeting(
        MeetingInfo(
            meeting_id=meeting_id,
            filename=file.filename or "upload",
            duration=0.0,
            provider=provider,
            created_at=datetime.utcnow(),
            status="processing",
        )
    )

    background_tasks.add_task(_pipeline, save_path, meeting_id, file.filename or "upload", provider)

    return {
        "meeting_id": meeting_id,
        "status": "processing",
        "provider": provider,
        "message": "Upload received. Transcription started in background.",
    }

@router.get("/status/{meeting_id}")
async def get_status(meeting_id: str):
    """Poll this endpoint to check transcription progress."""
    info = storage_service.get_meeting_info(meeting_id)
    if not info:
        raise HTTPException(404, "Meeting not found.")
    return {
        "meeting_id": info.meeting_id,
        "status": info.status,
        "filename": info.filename,
        "duration": info.duration,
        "provider": info.provider,
        "num_speakers": info.num_speakers,
    }

@router.get("/transcript/{meeting_id}", response_model=TranscriptionResult)
async def get_transcript(meeting_id: str):
    """Return the full transcript (only available when status = completed)."""
    info = storage_service.get_meeting_info(meeting_id)
    if not info:
        raise HTTPException(404, "Meeting not found.")
    if info.status == "processing":
        raise HTTPException(409, "Transcription still in progress.")
    if info.status == "failed":
        raise HTTPException(500, "Transcription failed. Please re-upload.")

    result = storage_service.load_meeting(meeting_id)
    if not result:
        raise HTTPException(404, "Transcript data not found.")
    return result


@router.get("/meetings", response_model=list[MeetingInfo])
async def list_meetings():
    """List all meetings ordered by newest first."""
    return storage_service.list_meetings()

@router.delete("/transcript/{meeting_id}", status_code=204)
async def delete_meeting(meeting_id: str):
    """Delete a meeting and all associated data (transcript, clusters, FAISS index)."""
    info = storage_service.get_meeting_info(meeting_id)
    if not info:
        raise HTTPException(404, "Meeting not found.")
    storage_service.delete_meeting(meeting_id)
    # Remove FAISS files
    from services.vector_store import _FAISS_DIR
    (_FAISS_DIR / f"{meeting_id}.index").unlink(missing_ok=True)
    (_FAISS_DIR / f"{meeting_id}_chunks.pkl").unlink(missing_ok=True)
    return JSONResponse(status_code=204, content=None)