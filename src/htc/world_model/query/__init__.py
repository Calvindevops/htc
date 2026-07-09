"""Query transformation — gBrain's "expansion model": transform a raw query
into better retrieval queries BEFORE hitting the store, retrieve for each,
and fuse via RRF. Pluggable and opt-in (each strategy costs an LLM call);
default "none" leaves zero-config retrieval byte-for-byte unchanged. Select
via `HTC_QUERY_TRANSFORM` or the `retrieve_with_transform` `strategy` arg.
"""

from __future__ import annotations

from .retrieve import STRATEGIES, retrieve_with_transform
from .transform import decompose, expand, hyde, multi_query

__all__ = [
    "STRATEGIES",
    "decompose",
    "expand",
    "hyde",
    "multi_query",
    "retrieve_with_transform",
]
