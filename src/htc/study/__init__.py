"""The correlation-study harness — validates whether the Agent-Ready score
predicts real task performance (agent-ladder x task-bank, blind grading,
Spearman + bootstrap CI). Grading is human-in-the-loop; this package builds
and analyzes the study, it does not grade automatically."""

from .bank import load_bank, save_bank
from .grading import ingest_grades, load_grading_sheet, make_grading_sheet, save_grading_sheet
from .model import AgentSpec, Attempt, Grade, Task
from .run import run_attempts
from .stats import bootstrap_ci, inter_rater_agreement, spearman, study_verdict

__all__ = [
    "AgentSpec",
    "Attempt",
    "Grade",
    "Task",
    "load_bank",
    "save_bank",
    "run_attempts",
    "make_grading_sheet",
    "save_grading_sheet",
    "load_grading_sheet",
    "ingest_grades",
    "spearman",
    "bootstrap_ci",
    "inter_rater_agreement",
    "study_verdict",
]
