"""
Thin HTTP client for the MeetRecall AI backend.
All methods return the parsed JSON dict (or a {"error": "..."} dict on failure).
"""
from __future__ import annotations

import os
from typing import Optional

import requests

_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


class APIClient:
    def __init__(self, base_url: str = _BACKEND_URL) -> None:
        self.base = base_url.rstrip("/")
        self._session = requests.Session()

    # ── Transcription ─────────────────────────────────────────────────────────

    def upload_file(self, file_obj, provider: str = "local") -> dict:
        """
        Upload an audio/video file to start transcription.
        provider: "local" (Whisper + Pyannote) | "assemblyai"
        """
        try:
            resp = self._session.post(
                f"{self.base}/api/transcribe",
                files={"file": (file_obj.name, file_obj, "application/octet-stream")},
                data={"provider": provider},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def get_status(self, meeting_id: str) -> dict:
        """Poll transcription status.
        Timeout is generous (60 s) because the single uvicorn worker may be
        busy loading HuggingFace models and temporarily slow to respond.
        A ReadTimeout is treated as still-processing so the caller keeps polling.
        """
        try:
            resp = self._session.get(f"{self.base}/api/status/{meeting_id}", timeout=60)
            if resp.status_code == 404:
                return {"error": "Meeting not found."}
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            return {"status": "processing"}
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def get_transcript(self, meeting_id: str) -> dict:
        """Retrieve full transcript.
        Returns {"status": "processing"} (HTTP 409 or Timeout) or
        {"status": "failed"} (HTTP 500) so callers can distinguish states
        without a separate status call.
        Timeout is 60 s to accommodate slow HuggingFace model responses.
        """
        try:
            resp = self._session.get(f"{self.base}/api/transcript/{meeting_id}", timeout=60)
            if resp.status_code == 404:
                return {"error": "Meeting not found."}
            if resp.status_code == 409:
                return {"status": "processing"}
            if resp.status_code == 500:
                detail = resp.json().get("detail", "Transcription failed.")
                return {"status": "failed", "error": detail}
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            return {"status": "processing"}
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def list_meetings(self) -> list:
        """List all meetings (newest first)."""
        try:
            resp = self._session.get(f"{self.base}/api/meetings", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return []

    def delete_meeting(self, meeting_id: str) -> bool:
        try:
            resp = self._session.delete(f"{self.base}/api/transcript/{meeting_id}", timeout=10)
            return resp.status_code == 204
        except requests.RequestException:
            return False

    # ── RAG Chat ──────────────────────────────────────────────────────────────

    def rag_chat(self, meeting_id: str, query: str, top_k: int = 3, history: list | None = None) -> dict:
        """Ask a question grounded in the meeting transcript.
        Pass history=[{"question":...,"answer":...}] for follow-up question support.
        """
        payload: dict = {"query": query, "top_k": top_k}
        if history:
            payload["history"] = history
        try:
            resp = self._session.post(
                f"{self.base}/api/rag/chat/{meeting_id}",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}

    # ── Clustering ────────────────────────────────────────────────────────────

    def run_clustering(
        self, meeting_id: str, method: str = "kmeans", n_clusters: int = 5
    ) -> dict:
        """Cluster transcript segments into topics."""
        try:
            resp = self._session.post(
                f"{self.base}/api/cluster/{meeting_id}",
                json={"method": method, "n_clusters": n_clusters},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def get_clusters(self, meeting_id: str) -> dict:
        """Get stored cluster results."""
        try:
            resp = self._session.get(f"{self.base}/api/cluster/{meeting_id}", timeout=10)
            if resp.status_code == 404:
                return {"error": "No cluster results yet."}
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}

    # ── Sentiment ─────────────────────────────────────────────────────────────

    def get_sentiment(self, meeting_id: str) -> dict:
        """Run per-speaker sentiment analysis (POST, computed on demand)."""
        try:
            resp = self._session.post(
                f"{self.base}/api/sentiment/{meeting_id}", timeout=360
            )
            if resp.status_code == 404:
                return {"error": "Meeting not found."}
            if resp.status_code == 409:
                return {"error": "Meeting not ready."}
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            return {"error": "Sentiment analysis timed out."}
        except requests.RequestException as exc:
            return {"error": str(exc)}

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        try:
            resp = self._session.get(f"{self.base}/health", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return {"status": "unreachable"}


api_client = APIClient()