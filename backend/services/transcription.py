"""
Transcription service - two providers:

  1. Local : openai/whisper-small (HuggingFace) + pyannote/speaker-diarization-3.1
  2. Cloud : AssemblyAI Universal-3 Pro (fast, accurate, includes speaker labels)

Both providers return Tuple[List[TranscriptSegment], str, float]
(segments, full_text, duration_seconds) so downstream services are provider-agnostic.
"""
from __future__ import annotations

import abc
import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import torch
from huggingface_hub import login
from transformers import pipeline as hf_pipeline

from models.schemas import TranscriptSegment

logger = logging.getLogger(__name__)

WHISPER_MODEL_ID     = "openai/whisper-small"
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"


# Utility 

def _check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH - install ffmpeg.")


async def convert_to_wav(input_path: Path) -> Path:
    """Convert any audio/video to 16 kHz mono WAV (required by Whisper and Pyannote)."""
    _check_ffmpeg()
    output_path = input_path.with_suffix(".wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")
    return output_path


# Base Provider 

class TranscriptionProvider(abc.ABC):
    @abc.abstractmethod
    async def transcribe(
        self, audio_path: Path
    ) -> Tuple[List[TranscriptSegment], str, float]:
        pass


# Local Whisper (internal — used by WhisperSmallWithDiarizationProvider) 

class LocalWhisperProvider:
    """Whisper-small ASR. Not exposed directly — used internally by the diarization provider."""

    _pipeline = None

    def __init__(self) -> None:
        self._device_int = 0 if torch.cuda.is_available() else -1
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        if LocalWhisperProvider._pipeline is not None:
            return
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_ID}")
        try:
            LocalWhisperProvider._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL_ID,
                chunk_length_s=30,
                device=self._device_int,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            logger.info("Whisper loaded.")
        except Exception as exc:
            logger.warning(f"Whisper load failed: {exc}. Clearing cache and retrying...")
            cache_dir = Path("/root/.cache/huggingface/hub/models--openai--whisper-small")
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            LocalWhisperProvider._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL_ID,
                chunk_length_s=30,
                device=self._device_int,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            logger.info("Whisper loaded on retry.")

    def run(self, wav_path: Path) -> Tuple[str, list]:
        result = LocalWhisperProvider._pipeline(
            str(wav_path),
            return_timestamps=True,
            generate_kwargs={"task": "transcribe"},
        )
        return result.get("text", ""), result.get("chunks", [])


# Pyannote Diarization 

class PyannoteDiarization:
    """Lazy-loaded pyannote speaker diarization pipeline. Shared across all requests."""

    _pipeline = None

    def __init__(self) -> None:
        self._hf_token: str = os.getenv("HUGGINGFACE_TOKEN", "")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enabled = bool(self._hf_token)

    def _initialize_pipeline(self) -> None:
        if PyannoteDiarization._pipeline is not None:
            return
        if not self.enabled:
            logger.warning(
                "HUGGINGFACE_TOKEN not set — diarization disabled. "
                "All segments will be labeled 'Speaker 1'."
            )
            return
        try:
            from pyannote.audio import Pipeline
            logger.info(f"Loading diarization model: {DIARIZATION_MODEL_ID}")
            login(token=self._hf_token, add_to_git_credential=False)
            PyannoteDiarization._pipeline = (
                Pipeline.from_pretrained(DIARIZATION_MODEL_ID).to(self._device)
            )
            logger.info("Pyannote diarization loaded successfully.")
        except Exception as exc:
            logger.warning(f"Pyannote failed to load: {exc}. Clearing cache and retrying...")
            cache_dir = Path(
                "/root/.cache/huggingface/hub/models--pyannote--speaker-diarization-3.1"
            )
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
                logger.warning("Pyannote cache cleared.")
            try:
                from pyannote.audio import Pipeline
                login(token=self._hf_token, add_to_git_credential=False)
                PyannoteDiarization._pipeline = (
                    Pipeline.from_pretrained(DIARIZATION_MODEL_ID).to(self._device)
                )
                logger.info("Pyannote loaded successfully on retry.")
            except Exception as exc2:
                logger.error(
                    f"Critical: Pyannote failed after cache clear: {exc2}\n"
                    "Ensure HUGGINGFACE_TOKEN is valid and model terms accepted at "
                    "https://hf.co/pyannote/speaker-diarization-3.1"
                )
                PyannoteDiarization._pipeline = None

    def diarize(self, wav_path: Path):
        """Run diarization on an already-converted WAV. Returns annotation or None."""
        self._initialize_pipeline()
        if PyannoteDiarization._pipeline is None:
            return None
        try:
            return PyannoteDiarization._pipeline(str(wav_path))
        except Exception as exc:
            logger.error(f"Diarization inference failed: {exc}")
            return None


# Local Provider: Whisper-Small + Pyannote 3.1 

class WhisperSmallWithDiarizationProvider(TranscriptionProvider):
    """
    Whisper-small (ASR) + Pyannote 3.1 (speaker diarization).
    This is the sole local provider — Whisper-only is not exposed externally.
    """

    def __init__(self) -> None:
        self.whisper  = LocalWhisperProvider()
        self.diarizer = PyannoteDiarization()

    async def transcribe(
        self, audio_path: Path
    ) -> Tuple[List[TranscriptSegment], str, float]:
        wav_path    = await convert_to_wav(audio_path)
        cleanup_wav = wav_path != audio_path

        try:
            loop = asyncio.get_running_loop()

            full_text, whisper_chunks = await loop.run_in_executor(
                None, self.whisper.run, wav_path
            )
            diarization = await loop.run_in_executor(
                None, self.diarizer.diarize, wav_path
            )

            segments = self._align(whisper_chunks, diarization)
            duration = segments[-1].end_time if segments else 0.0
            return segments, full_text, duration
        finally:
            if cleanup_wav and wav_path.exists():
                wav_path.unlink(missing_ok=True)

    @staticmethod
    def _align(whisper_chunks: list, diarization) -> List[TranscriptSegment]:
        """Assign speaker labels to Whisper chunks via maximum-overlap matching."""

        def _fallback(chunks: list) -> List[TranscriptSegment]:
            return [
                TranscriptSegment(
                    start_time=float((c.get("timestamp") or [0, 0])[0] or 0),
                    end_time=float((c.get("timestamp") or [0, 0])[1] or 0),
                    speaker="Speaker 1",
                    text=c["text"].strip(),
                )
                for c in chunks if c.get("text", "").strip()
            ]

        if not diarization:
            return _fallback(whisper_chunks)

        # Normalise annotation wrapper (pyannote may wrap the result)
        if hasattr(diarization, "speaker_diarization"):
            diarization = diarization.speaker_diarization
        elif hasattr(diarization, "annotation"):
            diarization = diarization.annotation

        try:
            diar_turns = list(diarization.itertracks(yield_label=True))
        except Exception:
            return _fallback(whisper_chunks)

        segments: List[TranscriptSegment] = []
        for chunk in whisper_chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            ts      = chunk.get("timestamp") or (0, 0)
            c_start = float(ts[0] or 0)
            c_end   = float(ts[1] or 0)

            best_speaker, max_overlap = "Speaker 1", 0.0
            for turn, _, spk in diar_turns:
                overlap = min(c_end, turn.end) - max(c_start, turn.start)
                if overlap > max_overlap:
                    max_overlap = overlap
                    # Pyannote labels: SPEAKER_00 → Speaker 1, SPEAKER_01 → Speaker 2
                    try:
                        spk_num = int(str(spk).split("_")[-1]) + 1
                    except (ValueError, IndexError):
                        spk_num = 1
                    best_speaker = f"Speaker {spk_num}"

            segments.append(
                TranscriptSegment(
                    start_time=c_start,
                    end_time=c_end,
                    speaker=best_speaker,
                    text=text,
                )
            )
        return segments


# Cloud Provider (AssemblyAI) 

class AssemblyAIProvider(TranscriptionProvider):
    """AssemblyAI Universal-3 Pro with automatic speaker diarization."""

    def __init__(self) -> None:
        self._api_key: str = os.getenv("ASSEMBLY_API_KEY", "")
        if not self._api_key:
            logger.warning("ASSEMBLY_API_KEY not set — AssemblyAI provider disabled.")

    async def transcribe(
        self, audio_path: Path
    ) -> Tuple[List[TranscriptSegment], str, float]:
        if not self._api_key:
            raise ValueError("ASSEMBLY_API_KEY is not configured.")

        import assemblyai as aai

        aai.settings.api_key = self._api_key
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(
            speech_models=[aai.SpeechModel.universal],
            speaker_labels=True,
            language_detection=True,
            punctuate=True,
            format_text=True,
            entity_detection=True,
        )

        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(
            None,
            lambda: transcriber.transcribe(str(audio_path), config),
        )

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI error: {transcript.error}")

        segments: List[TranscriptSegment] = []
        if transcript.utterances:
            for utt in transcript.utterances:
                segments.append(
                    TranscriptSegment(
                        start_time=utt.start / 1000.0,
                        end_time=utt.end / 1000.0,
                        speaker=f"Speaker {utt.speaker}",
                        text=utt.text,
                    )
                )
        else:
            segments.append(
                TranscriptSegment(
                    start_time=0.0,
                    end_time=float(transcript.audio_duration or 0),
                    speaker="Unknown",
                    text=transcript.text or "",
                )
            )

        full_text = transcript.text or ""
        duration  = float(transcript.audio_duration or 0)
        return segments, full_text, duration


# Orchestrator 

class TranscriptionOrchestrator:
    """
    Routes provider="local" / "assemblyai" to the correct provider.

    "local"      → WhisperSmallWithDiarizationProvider (Whisper-small + Pyannote 3.1)
    "assemblyai" → AssemblyAIProvider
    """

    def __init__(self) -> None:
        self._local      = WhisperSmallWithDiarizationProvider()
        self._assemblyai = AssemblyAIProvider()

    async def transcribe(
        self, audio_path: Path, provider: str = "local"
    ) -> Tuple[List[TranscriptSegment], str, float]:
        if provider == "assemblyai":
            return await self._assemblyai.transcribe(audio_path)
        return await self._local.transcribe(audio_path)


transcription_service = TranscriptionOrchestrator()