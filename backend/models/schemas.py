from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# Transcript 

class TranscriptSegment(BaseModel):
    start_time: float
    end_time: float
    speaker: str
    text: str


class TranscriptionResult(BaseModel):
    meeting_id: str
    filename: str
    duration: float
    segments: List[TranscriptSegment]
    full_text: str
    provider: str = "local"          
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "completed"


# Meeting Index 

class MeetingInfo(BaseModel):
    meeting_id: str
    filename: str
    duration: float
    provider: str = "local"
    created_at: datetime
    status: str                     
    num_speakers: int = 0
    num_segments: int = 0


# RAG 

class ChatTurn(BaseModel):
    """One Q&A turn for conversation memory (last N turns sent by client)."""
    question: str
    answer: str


class RAGChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    history: List[ChatTurn] = Field(
        default=[],
        description="Last N conversation turns for contextual follow-up questions.",
    )


class SourceChunk(BaseModel):
    text: str
    speaker: str
    start_time: float
    end_time: float
    score: float


class RAGChatResponse(BaseModel):
    """Structured meeting intelligence response (pure JSON, no HTML)."""
    meeting_id: str
    transcription_summary: str   
    key_actions: List[str]       
    keywords: List[str]          
    sources: List[SourceChunk]   


# Clustering 

class ClusterRequest(BaseModel):
    method: str = Field(default="kmeans", pattern="^(kmeans|hdbscan)$")
    n_clusters: int = Field(default=5, ge=2, le=20)


class ClusterTopic(BaseModel):
    cluster_id: int
    label: str            
    size: int
    speakers: List[str]
    sample_texts: List[str]
    time_range: Dict[str, float]


class ClusterResult(BaseModel):
    meeting_id: str
    method: str
    n_clusters_requested: int
    n_clusters_found: int
    topics: List[ClusterTopic]
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Sentiment 

class SpeakerSentiment(BaseModel):
    positive: float
    neutral: float
    negative: float

class SentimentResult(BaseModel):
    meeting_id: str
    speaker_sentiment: Dict[str, SpeakerSentiment]
    created_at: datetime = Field(default_factory=datetime.utcnow)