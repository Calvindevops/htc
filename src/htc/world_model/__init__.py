"""World-model: ingest arbitrary company sources (repo + docs) into citable chunks."""

from __future__ import annotations

from .retrieval import RetrievalPipeline, build_pipeline

__all__ = ["RetrievalPipeline", "build_pipeline"]
