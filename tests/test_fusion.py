"""Unit tests for the shared Reciprocal Rank Fusion core."""

import pytest

from htc.world_model.fusion import RRF_K, reciprocal_rank_fusion


def test_boosts_item_ranked_high_in_both_lists():
    fused = reciprocal_rank_fusion(
        [["a", "b", "c"], ["b", "a", "c"]],
        key=lambda id_: id_,
    )
    ids = [id_ for id_, _ in fused]
    # "a" and "b" both appear near the top of both rankings, so they should
    # outscore "c", which only ever appears last.
    assert ids[0] in ("a", "b")
    assert ids[1] in ("a", "b")
    assert ids[2] == "c"


def test_deterministic_tiebreak_on_key():
    # "a" and "b" swap rank between the two lists, so their fused scores tie
    # exactly; the deterministic tiebreak (ascending key) must decide order.
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], key=lambda id_: id_)
    assert [id_ for id_, _ in fused] == ["a", "b"]


def test_single_list_passthrough_preserves_order():
    fused = reciprocal_rank_fusion([["x", "y", "z"]], key=lambda id_: id_)
    assert [id_ for id_, _ in fused] == ["x", "y", "z"]


def test_k60_weighting():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]], key=lambda id_: id_))
    assert RRF_K == 60
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)


def test_repeatedly_calling_is_deterministic():
    lists = [["a", "b", "c"], ["c", "b", "a"], ["b", "c", "a"]]
    first = reciprocal_rank_fusion(lists, key=lambda id_: id_)
    second = reciprocal_rank_fusion(lists, key=lambda id_: id_)
    assert first == second
