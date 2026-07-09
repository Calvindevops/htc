"""Scorecard: score math, category buckets, badge thresholds, compare deltas."""

from __future__ import annotations

from htc.evaluation.runner import EvalResult, ItemResult
from htc.evaluation.scorecard import badge_url, render_compare, render_scorecard, scorecard_markdown
from htc.goldens.generator import Golden


def _result(verdicts_by_category: dict[str, list[str]]) -> EvalResult:
    items = []
    for category, verdicts in verdicts_by_category.items():
        for index, verdict in enumerate(verdicts):
            items.append(
                ItemResult(
                    golden=Golden(
                        question=f"{category} q{index}?",
                        answer="a",
                        artifact="f.py",
                        category=category,
                        difficulty=1,
                    ),
                    agent_answer="a",
                    verdict=verdict,
                    reason="r",
                )
            )
    return EvalResult(root="/repo", agent="builtin:test", items=items)


class TestScore:
    def test_all_correct_is_100(self):
        assert _result({"config": ["correct", "correct"]}).score == 100.0

    def test_partial_counts_half(self):
        assert _result({"config": ["correct", "partial"]}).score == 75.0

    def test_all_wrong_is_0(self):
        assert _result({"config": ["wrong", "wrong"]}).score == 0.0

    def test_empty_is_0(self):
        assert EvalResult(root="/r", agent="a", items=[]).score == 0.0

    def test_by_category(self):
        result = _result({"config": ["correct"], "ops": ["wrong"]})
        assert result.by_category() == {"config": 100.0, "ops": 0.0}


class TestBadge:
    def test_thresholds(self):
        assert "brightgreen" in badge_url(90)
        assert badge_url(75).endswith("green")
        assert "yellow" in badge_url(60)
        assert "red" in badge_url(30)


class TestRendering:
    def test_scorecard_shows_score_and_gaps(self):
        text = render_scorecard(_result({"config": ["correct", "wrong"]}))
        assert "50.0/100" in text
        assert "htc onboard" in text

    def test_markdown_has_badge_and_table(self):
        md = scorecard_markdown(_result({"ops": ["partial"]}))
        assert "img.shields.io" in md
        assert "| ops | 50.0 |" in md

    def test_compare_shows_delta(self):
        before = _result({"config": ["wrong", "wrong"]})
        after = _result({"config": ["correct", "correct"]})
        text = render_compare(before, after)
        assert "0.0 → 100.0" in text
        assert "+100.0" in text
