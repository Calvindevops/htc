"""Knowledge graph: a self-wiring graph of code symbols, files, modules,
proper nouns, and salient terms — extracted with ZERO LLM calls (deterministic
regex/frequency heuristics, mirroring gBrain's approach), persisted
HTC-native, and used as an additional retrieval signal."""

from .extract import extract
from .graph import KnowledgeGraph, build_graph, graph_json_path
from .model import Entity, Relation

__all__ = [
    "Entity",
    "KnowledgeGraph",
    "Relation",
    "build_graph",
    "extract",
    "graph_json_path",
]
