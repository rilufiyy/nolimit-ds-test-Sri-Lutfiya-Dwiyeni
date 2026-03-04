"""
Simple JSON-file persistence - replaces PostgreSQL for portability.

Layout:
  data/
  +-- meetings.json                  <- ordered index of all meetings
  +-- meetings/
      +-- {meeting_id}.json          <- full TranscriptionResult
      +-- {meeting_id}_clusters.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from models.schemas import ClusterResult, MeetingInfo, TranscriptionResult

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data")
_MEETINGS_DIR = _DATA_DIR / "meetings"
_INDEX_PATH = _DATA_DIR / "meetings.json"

for _p in (_DATA_DIR, _MEETINGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

if not _INDEX_PATH.exists():
    _INDEX_PATH.write_text("[]", encoding="utf-8")


def _dt_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serialisable")


class StorageService:

    # Index helpers

    def _read_index(self) -> list:
        try:
            return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write_index(self, data: list) -> None:
        _INDEX_PATH.write_text(
            json.dumps(data, indent=2, default=_dt_serial),
            encoding="utf-8",
        )

    def _upsert_index(self, info: MeetingInfo) -> None:
        index = self._read_index()
        entry = json.loads(json.dumps(info.model_dump(), default=_dt_serial))
        for i, m in enumerate(index):
            if m["meeting_id"] == info.meeting_id:
                index[i] = entry
                self._write_index(index)
                return
        index.insert(0, entry)   # newest first
        self._write_index(index)

    # Public API

    def list_meetings(self) -> List[MeetingInfo]:
        return [MeetingInfo(**m) for m in self._read_index()]

    def create_meeting(self, info: MeetingInfo) -> None:
        """Register a new meeting (status=processing placeholder)."""
        self._upsert_index(info)

    def save_meeting(self, result: TranscriptionResult) -> None:
        """Persist completed transcription result and update index."""
        data = json.loads(json.dumps(result.model_dump(), default=_dt_serial))
        (_MEETINGS_DIR / f"{result.meeting_id}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        info = MeetingInfo(
            meeting_id=result.meeting_id,
            filename=result.filename,
            duration=result.duration,
            provider=result.provider,
            created_at=result.created_at,
            status=result.status,
            num_speakers=len({s.speaker for s in result.segments}),
            num_segments=len(result.segments),
        )
        self._upsert_index(info)
        logger.info(f"Saved meeting {result.meeting_id}")

    def load_meeting(self, meeting_id: str) -> Optional[TranscriptionResult]:
        path = _MEETINGS_DIR / f"{meeting_id}.json"
        if not path.exists():
            return None
        return TranscriptionResult(**json.loads(path.read_text(encoding="utf-8")))

    def get_meeting_info(self, meeting_id: str) -> Optional[MeetingInfo]:
        for m in self._read_index():
            if m["meeting_id"] == meeting_id:
                return MeetingInfo(**m)
        return None

    def update_status(self, meeting_id: str, status: str) -> None:
        index = self._read_index()
        for m in index:
            if m["meeting_id"] == meeting_id:
                m["status"] = status
                break
        self._write_index(index)

    def delete_meeting(self, meeting_id: str) -> None:
        path = _MEETINGS_DIR / f"{meeting_id}.json"
        path.unlink(missing_ok=True)
        (_MEETINGS_DIR / f"{meeting_id}_clusters.json").unlink(missing_ok=True)
        index = [m for m in self._read_index() if m["meeting_id"] != meeting_id]
        self._write_index(index)

    def save_clusters(self, meeting_id: str, result: ClusterResult) -> None:
        data = json.loads(json.dumps(result.model_dump(), default=_dt_serial))
        (_MEETINGS_DIR / f"{meeting_id}_clusters.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def load_clusters(self, meeting_id: str) -> Optional[ClusterResult]:
        path = _MEETINGS_DIR / f"{meeting_id}_clusters.json"
        if not path.exists():
            return None
        return ClusterResult(**json.loads(path.read_text(encoding="utf-8")))


storage_service = StorageService()