"""
Sentiment analysis service — runs 100% locally via HuggingFace.

Model : cardiffnlp/twitter-roberta-base-sentiment
Labels: LABEL_0 = Negative · LABEL_1 = Neutral · LABEL_2 = Positive

Pipeline processes segments per speaker in batches of 32,
then averages probability scores to produce a distribution.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from models.schemas import SentimentResult, SpeakerSentiment, TranscriptSegment

logger = logging.getLogger(__name__)

_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment"
_LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
_BATCH = 32


class SentimentService:
    """Per-speaker sentiment aggregation using a local RoBERTa model."""

    _pipeline = None

    def _load(self) -> None:
        if SentimentService._pipeline is not None:
            return
        import torch
        from transformers import pipeline as hf_pipeline

        logger.info(f"Loading sentiment model: {_MODEL_ID}")
        device = 0 if torch.cuda.is_available() else -1
        SentimentService._pipeline = hf_pipeline(
            "text-classification",
            model=_MODEL_ID,
            tokenizer=_MODEL_ID,
            device=device,
            truncation=True,
            max_length=128,
        )
        logger.info("Sentiment model loaded.")

    # Public API 

    def analyse(self, segments: List[TranscriptSegment]) -> SentimentResult:
        """
        Compute per-speaker average sentiment distribution.

        Args:
            segments : transcript segments (speaker + text).

        Returns:
            SentimentResult with speaker_sentiment mapping.
        """
        self._load()
        pipe = SentimentService._pipeline

        # Group non-empty texts by speaker
        by_speaker: Dict[str, List[str]] = defaultdict(list)
        for seg in segments:
            txt = seg.text.strip()
            if txt:
                by_speaker[seg.speaker].append(txt)

        speaker_sentiment: Dict[str, SpeakerSentiment] = {}

        for speaker, texts in by_speaker.items():
            agg = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            count = 0

            for i in range(0, len(texts), _BATCH):
                batch = texts[i : i + _BATCH]
                outputs = pipe(batch, top_k=None)
                for scores_list in outputs:
                    for item in scores_list:
                        key = _LABEL_MAP.get(item["label"], "neutral")
                        agg[key] += item["score"]
                    count += 1

            if count:
                speaker_sentiment[speaker] = SpeakerSentiment(
                    positive=round(agg["positive"] / count, 4),
                    neutral=round(agg["neutral"] / count, 4),
                    negative=round(agg["negative"] / count, 4),
                )

        return SentimentResult(
            meeting_id="",  # filled by caller
            speaker_sentiment=speaker_sentiment,
        )


sentiment_service = SentimentService()