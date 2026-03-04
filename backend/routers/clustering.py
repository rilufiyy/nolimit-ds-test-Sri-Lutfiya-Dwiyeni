"""
Topic clustering router.

Endpoints:
  POST /api/cluster/{meeting_id} - run clustering and store result
  GET  /api/cluster/{meeting_id} - retrieve stored cluster result
"""
from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from models.schemas import ClusterRequest, ClusterResult
from services.storage import storage_service
from services.topic_cluster import topic_clusterer
from services.vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cluster", tags=["Topic Clustering"])


def _require_completed(meeting_id: str) -> None:
    info = storage_service.get_meeting_info(meeting_id)
    if not info:
        raise HTTPException(404, "Meeting not found.")
    if info.status != "completed":
        raise HTTPException(409, f"Meeting not ready. Current status: {info.status}")


@router.post("/{meeting_id}", response_model=ClusterResult)
async def run_clustering(meeting_id: str, request: ClusterRequest):
    """
    Cluster transcript segments into thematic topics.

    - **method**: `kmeans` (default, deterministic) or `hdbscan` (auto-discovers clusters).
    - **n_clusters**: target number of clusters for KMeans (ignored for HDBSCAN).

    Returns cluster topics with a representative label, sample texts, speakers,
    and the time range each topic spans in the meeting.
    """
    _require_completed(meeting_id)

    # Load full transcript (for segment objects)
    result = storage_service.load_meeting(meeting_id)
    if not result:
        raise HTTPException(404, "Transcript data not found.")
    if not result.segments:
        raise HTTPException(400, "No transcript segments available to cluster.")

    # Get embeddings from FAISS (avoids re-encoding)
    emb_data = vector_store.get_all_embeddings(meeting_id)
    if emb_data is None:
        # Lazily build FAISS index
        from services.embeddings import embedding_service

        texts = [s.text for s in result.segments]
        meta = [
            {"speaker": s.speaker, "start_time": s.start_time, "end_time": s.end_time}
            for s in result.segments
        ]
        vector_store.index_meeting(meeting_id, texts, meta)
        emb_data = vector_store.get_all_embeddings(meeting_id)

    embeddings, _ = emb_data
    segments = result.segments

    # Align lengths (FAISS stores one vector per segment)
    min_len = min(len(segments), len(embeddings))
    segments = segments[:min_len]
    embeddings = embeddings[:min_len]

    try:
        topics = topic_clusterer.cluster(
            segments=segments,
            embeddings=embeddings,
            method=request.method,
            n_clusters=request.n_clusters,
        )
    except Exception as exc:
        logger.error(f"Clustering failed for {meeting_id}: {exc}")
        raise HTTPException(500, f"Clustering error: {exc}")

    cluster_result = ClusterResult(
        meeting_id=meeting_id,
        method=request.method,
        n_clusters_requested=request.n_clusters,
        n_clusters_found=len(topics),
        topics=topics,
    )
    storage_service.save_clusters(meeting_id, cluster_result)
    return cluster_result


@router.get("/{meeting_id}", response_model=ClusterResult)
async def get_clusters(meeting_id: str):
    """Retrieve the most recent cluster result for a meeting (if available)."""
    _require_completed(meeting_id)
    result = storage_service.load_clusters(meeting_id)
    if not result:
        raise HTTPException(
            404,
            "No cluster results found for this meeting. "
            "Run POST /api/cluster/{meeting_id} first.",
        )
    return result