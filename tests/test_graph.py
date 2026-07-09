"""Knowledge graph: deterministic entity/relation extraction (ZERO LLM
calls, no network, no randomness), KnowledgeGraph container, and the
optional graph-boost retrieval signal in LocalMemoryStore.search."""

from __future__ import annotations

from htc.world_model.graph import Entity, KnowledgeGraph, Relation, build_graph, extract
from htc.world_model.ingest.model import SourceChunk
from htc.world_model.memory.local import LocalMemoryStore


def _chunk(id_: str, source_path: str, text: str, kind: str = "code") -> SourceChunk:
    return SourceChunk(
        id=id_, source_path=source_path, kind=kind, text=text, start_char=0, end_char=len(text)
    )


_BUILD_CHUNKS = [
    _chunk(
        "a",
        "src/build.py",
        "def build_memory(sources):\n"
        "    return sources\n\n"
        "class MemoryStore:\n"
        "    pass\n"
        "def build_memory_again(sources):\n"
        "    return build_memory(sources)\n",
    ),
    _chunk(
        "b",
        "src/cli.py",
        "from src.build import build_memory\n\n"
        "build_memory([])\n"
        "Agent Ready score matters a lot for Agent Ready teams.\n",
    ),
]


class TestExtract:
    def test_finds_code_symbols(self):
        entities, _ = extract(_BUILD_CHUNKS)
        symbols = {e.name for e in entities if e.kind == "symbol"}
        assert "build_memory" in symbols
        assert "MemoryStore" in symbols

    def test_finds_file_entities(self):
        entities, _ = extract(_BUILD_CHUNKS)
        files = {e.name for e in entities if e.kind == "file"}
        assert files == {"src/build.py", "src/cli.py"}

    def test_finds_proper_nouns(self):
        entities, _ = extract(_BUILD_CHUNKS)
        nouns = {e.name for e in entities if e.kind == "proper_noun"}
        assert "Agent Ready" in nouns

    def test_contains_relation_links_file_to_its_symbol(self):
        entities, relations = extract(_BUILD_CHUNKS)
        by_id = {e.id: e for e in entities}
        contains = [r for r in relations if r.kind == "contains"]
        assert contains
        for relation in contains:
            assert by_id[relation.source_id].kind == "file"
            assert by_id[relation.target_id].kind == "symbol"
        symbol_targets = {by_id[r.target_id].name for r in contains}
        assert "build_memory" in symbol_targets

    def test_co_occurs_relation_created(self):
        _, relations = extract(_BUILD_CHUNKS)
        assert any(r.kind == "co_occurs" for r in relations)

    def test_references_relation_for_cross_file_symbol_use(self):
        entities, relations = extract(_BUILD_CHUNKS)
        by_id = {e.id: e for e in entities}
        references = [r for r in relations if r.kind == "references"]
        assert references
        assert any(
            by_id[r.source_id].name == "build_memory" and by_id[r.target_id].name == "src/cli.py"
            for r in references
        )

    def test_extraction_is_deterministic(self):
        entities1, relations1 = extract(_BUILD_CHUNKS)
        entities2, relations2 = extract(list(reversed(_BUILD_CHUNKS)))
        assert entities1 == entities2
        assert relations1 == relations2

    def test_no_llm_import(self):
        """Zero LLM calls: extract.py/graph.py must not import htc.llm."""
        import importlib

        for module_name in ("htc.world_model.graph.extract", "htc.world_model.graph.graph"):
            module = importlib.import_module(module_name)
            with open(module.__file__) as f:
                source = f.read()
            assert ".llm import" not in source
            assert "from .llm" not in source


class TestKnowledgeGraph:
    def test_add_and_top_entities(self):
        graph = KnowledgeGraph()
        graph.add_entities(
            [
                Entity(id="term:a", name="a", kind="term", mentions=5),
                Entity(id="term:b", name="b", kind="term", mentions=1),
            ]
        )
        top = graph.top_entities(1)
        assert [e.id for e in top] == ["term:a"]

    def test_neighbors_ordered_by_weight(self):
        graph = KnowledgeGraph()
        graph.add_entities(
            [
                Entity(id="a", name="a", kind="term"),
                Entity(id="b", name="b", kind="term"),
                Entity(id="c", name="c", kind="term"),
            ]
        )
        graph.add_relations(
            [
                Relation("a", "b", "co_occurs", weight=1),
                Relation("a", "c", "co_occurs", weight=5),
            ]
        )
        neighbors = graph.neighbors("a", k=2)
        assert [e.id for e in neighbors] == ["c", "b"]

    def test_subgraph_for_query_returns_matched_plus_neighbors(self):
        entities, relations = extract(_BUILD_CHUNKS)
        graph = KnowledgeGraph()
        graph.add_entities(entities)
        graph.add_relations(relations)
        result = graph.subgraph_for_query("build_memory", k=10)
        names = {e.name for e in result}
        assert "build_memory" in names
        # its neighbors (e.g. the file that contains/references it) should
        # be pulled in too.
        assert len(result) > 1

    def test_save_and_load_round_trip(self, tmp_path):
        entities, relations = extract(_BUILD_CHUNKS)
        graph = KnowledgeGraph()
        graph.add_entities(entities)
        graph.add_relations(relations)
        graph.save(tmp_path)

        reloaded = KnowledgeGraph.load(tmp_path)
        assert reloaded.entities() == graph.entities()
        assert reloaded.relations() == graph.relations()

    def test_load_missing_graph_returns_empty(self, tmp_path):
        graph = KnowledgeGraph.load(tmp_path)
        assert graph.entities() == []
        assert graph.relations() == []

    def test_to_mermaid_produces_fenced_block(self):
        entities, relations = extract(_BUILD_CHUNKS)
        graph = KnowledgeGraph()
        graph.add_entities(entities)
        graph.add_relations(relations)
        rendered = graph.to_mermaid()
        assert rendered.startswith("```mermaid\n")
        assert rendered.rstrip().endswith("```")
        assert "graph LR" in rendered

    def test_build_graph_convenience_persists(self, tmp_path):
        graph = build_graph(_BUILD_CHUNKS, tmp_path)
        assert (tmp_path / ".htc" / "graph" / "graph.json").is_file()
        assert graph.entities()


class TestGraphBoostRetrieval:
    def test_no_op_when_graph_not_passed(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        store.add_chunks(
            [
                _chunk("x", "shipping.md", "Shipping takes 5 to 7 business days.", kind="docs"),
                _chunk("y", "refunds.md", "Refund policy covers unopened items.", kind="docs"),
            ]
        )
        without_graph = store.search("shipping refund policy", k=2)
        with_none_graph = store.search("shipping refund policy", k=2, graph=None)
        assert without_graph == with_none_graph

    def test_graph_boost_changes_ranking(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        store.add_chunks(
            [
                _chunk(
                    "x",
                    "shipping.md",
                    "Shipping takes 5 to 7 business days for most orders nationwide.",
                    kind="docs",
                ),
                _chunk(
                    "y",
                    "widget.md",
                    "The Widget Corp product line ships quickly to every region.",
                    kind="docs",
                ),
            ]
        )
        graph = KnowledgeGraph()
        graph.add_entities(
            [
                Entity(id="file:widget-md", name="widget.md", kind="file", mentions=1),
                Entity(
                    id="proper_noun:widget-corp", name="Widget Corp", kind="proper_noun", mentions=1
                ),
            ]
        )
        graph.add_relations(
            [Relation("file:widget-md", "proper_noun:widget-corp", "co_occurs", weight=3)]
        )

        baseline = store.search("orders", k=2)
        boosted = store.search("orders widget corp", k=2, graph=graph)
        assert boosted
        assert boosted[0].chunk.source_path == "widget.md"
        assert baseline[0].chunk.source_path == "shipping.md"
