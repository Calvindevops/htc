"""Correlation-study harness: spearman/bootstrap correctness, blind grading
round-trip, task-bank round-trip, and the end-to-end synthetic verdict."""

from __future__ import annotations

from htc.study.bank import load_bank, save_bank
from htc.study.grading import ingest_grades, make_grading_sheet
from htc.study.model import Attempt, Grade, Task
from htc.study.stats import bootstrap_ci, spearman, study_verdict


def _task(**overrides):
    base = dict(id="t1", prompt="do the thing", category="ops", provenance="PR #1")
    base.update(overrides)
    return Task(**base)


class TestSpearman:
    def test_perfect_positive_correlation(self):
        assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == 1.0

    def test_perfect_negative_correlation(self):
        assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == -1.0

    def test_zero_correlation(self):
        # a permutation of ranks chosen so the rank-covariance is exactly zero
        assert spearman([1, 2, 3, 4], [2, 4, 1, 3]) == 0.0

    def test_degenerate_input_returns_zero(self):
        assert spearman([1], [1]) == 0.0
        assert spearman([1, 2], [1]) == 0.0


class TestBootstrapCi:
    def test_deterministic_with_fixed_seed(self):
        pairs = [(1, 10), (2, 20), (3, 15), (4, 40), (5, 45)]
        first = bootstrap_ci(pairs, spearman, n=200, seed=42)
        second = bootstrap_ci(pairs, spearman, n=200, seed=42)
        assert first == second

    def test_different_seeds_can_differ(self):
        pairs = [(1, 10), (2, 5), (3, 40), (4, 2), (5, 45)]
        a = bootstrap_ci(pairs, spearman, n=200, seed=1)
        b = bootstrap_ci(pairs, spearman, n=200, seed=2)
        # not asserting inequality (could coincide) — just that both are valid ranges
        assert a[0] <= a[1]
        assert b[0] <= b[1]

    def test_empty_pairs_returns_zero_zero(self):
        assert bootstrap_ci([], spearman, n=100, seed=0) == (0.0, 0.0)


class TestMakeGradingSheet:
    def test_deterministic_with_seed(self):
        tasks = [_task(id="t1"), _task(id="t2"), _task(id="t3")]
        attempts = [
            Attempt(task_id="t1", agent_id="base", output="a1"),
            Attempt(task_id="t2", agent_id="base", output="a2"),
            Attempt(task_id="t3", agent_id="full", output="a3"),
        ]
        sheet_a = make_grading_sheet(attempts, tasks, seed=7)
        sheet_b = make_grading_sheet(attempts, tasks, seed=7)
        assert sheet_a == sheet_b

    def test_strips_agent_identity_from_blind_view(self):
        tasks = [_task(id="t1")]
        attempts = [Attempt(task_id="t1", agent_id="base", output="the answer")]
        sheet = make_grading_sheet(attempts, tasks, seed=1)
        row = sheet[0]
        # the human-facing fields carry no agent label anywhere in their values
        assert row["blind_id"] == "blind-000"
        assert row["task_prompt"] == "do the thing"
        assert row["output"] == "the answer"
        assert "base" not in row["blind_id"]
        assert "base" not in row["task_prompt"]

    def test_shuffles_order_relative_to_input(self):
        tasks = [_task(id=f"t{i}") for i in range(8)]
        attempts = [Attempt(task_id=t.id, agent_id="base", output=t.id) for t in tasks]
        sheet = make_grading_sheet(attempts, tasks, seed=3)
        order = [row["task_id"] for row in sheet]
        assert order != [t.id for t in tasks]
        assert sorted(order) == sorted(t.id for t in tasks)


class TestIngestGrades:
    def test_maps_blind_ids_back_to_task_and_agent(self):
        tasks = [_task(id="t1"), _task(id="t2")]
        attempts = [
            Attempt(task_id="t1", agent_id="base", output="a1"),
            Attempt(task_id="t2", agent_id="full", output="a2"),
        ]
        sheet = make_grading_sheet(attempts, tasks, seed=5)
        filled = {row["blind_id"]: 3 for row in sheet}
        grades = ingest_grades(sheet, filled, grader_id="alice")
        assert len(grades) == 2
        by_task = {g.task_id: g for g in grades}
        assert by_task["t1"].agent_id == "base"
        assert by_task["t1"].grader_id == "alice"
        assert by_task["t1"].score == 3
        assert by_task["t2"].agent_id == "full"

    def test_unknown_blind_id_raises(self):
        sheet = make_grading_sheet(
            [Attempt(task_id="t1", agent_id="base", output="a1")], [_task(id="t1")], seed=0
        )
        try:
            ingest_grades(sheet, {"blind-999": 2}, grader_id="bob")
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestStudyVerdict:
    def test_passes_on_strongly_correlated_synthetic_data(self):
        score_by_agent = {"base": 20.0, "partial": 50.0, "full": 90.0, "human": 95.0}
        grades = []
        means = {"base": 1.0, "partial": 2.0, "full": 3.5, "human": 4.0}
        for agent_id, mean_score in means.items():
            for task_i in range(6):
                for grader_id in ("g1", "g2"):
                    grades.append(
                        Grade(
                            task_id=f"t{task_i}",
                            agent_id=agent_id,
                            grader_id=grader_id,
                            score=int(mean_score),
                        )
                    )
        verdict = study_verdict(score_by_agent, grades, seed=0)
        assert verdict["passed"] is True
        assert verdict["rho"] >= 0.6
        assert verdict["n_points"] == 4
        assert verdict["n_graders"] == 2

    def test_fails_on_uncorrelated_synthetic_data(self):
        score_by_agent = {"base": 90.0, "partial": 20.0, "full": 95.0, "human": 10.0}
        grades = []
        means = {"base": 1, "partial": 4, "full": 1, "human": 4}
        for agent_id, score in means.items():
            for task_i in range(4):
                grades.append(
                    Grade(task_id=f"t{task_i}", agent_id=agent_id, grader_id="g1", score=score)
                )
        verdict = study_verdict(score_by_agent, grades, seed=0)
        assert verdict["passed"] is False


class TestBankRoundTrip:
    def test_save_then_load(self, tmp_path):
        tasks = [_task(id="t1"), _task(id="t2", category="config")]
        path = save_bank(tasks, tmp_path / "bank.json")
        loaded = load_bank(path)
        assert loaded == tasks

    def test_warns_when_fewer_than_recommended(self, tmp_path, capsys):
        tasks = [_task(id="t1")]
        path = save_bank(tasks, tmp_path / "bank.json")
        load_bank(path)
        assert "fewer than the recommended" in capsys.readouterr().err
