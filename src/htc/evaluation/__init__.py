"""Evaluation — score an agent against the repo's golden Q&A."""

from .runner import EvalResult, ItemResult, run_eval
from .scorecard import render_compare, render_scorecard, scorecard_markdown

__all__ = [
    "EvalResult",
    "ItemResult",
    "run_eval",
    "render_scorecard",
    "render_compare",
    "scorecard_markdown",
]
