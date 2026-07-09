"""Generate repo-specific golden Q&A — the knowledge exam an agent must pass.

Files are sampled weighted by git churn (frequently-changed files hold the
knowledge that matters); repos without git history fall back to uniform
sampling. Every generated item is validated against the repo: an item is kept
only if its `artifact` path actually exists. Output is `.htc/goldens.json`.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ..llm import LLMError, complete, extract_json

CATEGORIES = ("architecture", "config", "behavior", "ops")
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    "dist",
    "build",
    ".htc",
}
SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".mp4",
    ".woff",
    ".woff2",
    ".ttf",
    ".lock",
    ".map",
    ".pyc",
    ".svg",
    ".pdf",
}
MAX_FILE_CHARS = 6_000
FILES_PER_BATCH = 4

# Paths matching any of these are test artifacts: down-weighted during sampling
# and hard-rejected as a golden's `artifact` in _validate.
TEST_MARKERS = ("tests/", "test_", "_test.", "/spec/", ".spec.", "__snapshots__")

# Secret-bearing files are never read into a prompt — their contents would leak
# into generated goldens. Excluded from sampling entirely (belt-and-suspenders
# for enterprise repos; .env* and common key/credential files).
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".keystore", ".jks"}
SECRET_NAMES = {"credentials", "secrets", ".npmrc", ".pypirc", ".netrc", ".htpasswd"}
SECRET_STEMS = ("id_rsa", "id_ed25519", "id_dsa", "id_ecdsa")


def _is_secret_file(path: Path) -> bool:
    """True if `path` looks like it carries secrets (.env*, keys, credentials)."""
    name = path.name.lower()
    if name.startswith(".env"):
        return True
    if path.suffix.lower() in SECRET_SUFFIXES:
        return True
    if name in SECRET_NAMES:
        return True
    return any(name.startswith(stem) for stem in SECRET_STEMS)


GENERATION_SYSTEM = """You write evaluation questions that test whether an AI agent \
genuinely knows a specific codebase. You are given real file contents.

Every question MUST probe exactly ONE of:
- a DECISION — why this approach and not the obvious alternative
- a CONSTRAINT / INVARIANT — what must stay true or the system breaks
- a FAILURE MODE — what happens, or what the fallback is, when X fails
- a DATA-FLOW / WIRING fact — where a value comes from, what calls what

BANNED — reject these even if true:
- pure lookup/naming questions ("what is X called", "what does module Y export", \
"what is the default/threshold/constant for Z")
- anything answerable by reading a single literal without understanding why it's there
- anything a test file alone would answer

Grounding rule: `artifact` MUST be an implementation/source or config file — under \
src/, lib/, app/, or a root-level config file. NEVER a test file, fixture, or snapshot.

The answer must be short, factual, and verifiable against the files shown.
Categories: architecture (structure/data flow), config (env/settings/deps), \
behavior (what the code does in a scenario), ops (deploy/run/tooling).
difficulty: 1 = findable in one file, 2 = needs connecting two facts, \
3 = needs real understanding of the design.

Reply with ONLY a JSON array of objects: \
{"question": str, "answer": str, "artifact": str, "category": str, "difficulty": int}"""


@dataclass(frozen=True)
class Golden:
    """One golden Q&A item."""

    question: str
    answer: str
    artifact: str
    category: str
    difficulty: int


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if _is_secret_file(path):
            continue
        try:
            if path.stat().st_size > 200_000:
                continue
        except OSError:
            continue
        files.append(path)
    return files


def _churn_weights(root: Path, files: list[Path]) -> dict[Path, int]:
    """Commit-touch counts per file from git log; empty dict if not a git repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "log", "--name-only", "--pretty=format:", "-500"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if out.returncode != 0:
        return {}
    counts = Counter(line.strip() for line in out.stdout.splitlines() if line.strip())
    known = {f.relative_to(root).as_posix(): f for f in files}
    return {known[rel]: n for rel, n in counts.items() if rel in known}


def _is_test_file(path: Path, root: Path) -> bool:
    """True if `path` is a test/fixture/snapshot file (down-weighted, never grounding)."""
    rel = path.relative_to(root).as_posix()
    return any(marker in rel for marker in TEST_MARKERS)


def _sample_files(root: Path, count: int, rng: random.Random) -> list[Path]:
    files = _iter_files(root)
    if not files:
        return []
    weights = _churn_weights(root, files)
    if weights:
        # Churn-weighted, but every file keeps a base weight so cold files can appear.
        # Test files stay in the pool (they still hold useful context) but at 1/5
        # weight — goldens should mostly come from implementation, not test code.
        pool = files
        w = [(1 + weights.get(f, 0)) * (0.2 if _is_test_file(f, root) else 1.0) for f in pool]
        picked: list[Path] = []
        remaining = list(zip(pool, w))
        for _ in range(min(count, len(pool))):
            total = sum(wt for _, wt in remaining)
            r = rng.uniform(0, total)
            acc = 0.0
            for i, (f, wt) in enumerate(remaining):
                acc += wt
                if r <= acc:
                    picked.append(f)
                    remaining.pop(i)
                    break
        return picked
    return rng.sample(files, min(count, len(files)))


def _read_clipped(path: Path) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return text if len(text) <= MAX_FILE_CHARS else text[:MAX_FILE_CHARS] + "\n...[truncated]"


def _validate(items: list[dict], root: Path) -> list[Golden]:
    goldens: list[Golden] = []
    for item in items:
        try:
            golden = Golden(
                question=str(item["question"]).strip(),
                answer=str(item["answer"]).strip(),
                artifact=str(item["artifact"]).strip().lstrip("/"),
                category=str(item["category"]).strip(),
                difficulty=int(item["difficulty"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not golden.question or not golden.answer:
            continue
        if golden.category not in CATEGORIES:
            continue
        if golden.difficulty not in (1, 2, 3):
            continue
        # The grounding gate: the cited artifact must exist in the repo.
        if not (root / golden.artifact).is_file():
            continue
        # Hard guarantee: never ground a golden in a test/fixture/snapshot file.
        if any(marker in golden.artifact for marker in TEST_MARKERS):
            continue
        goldens.append(golden)
    return goldens


def _balance_hint(counts: Counter, target_total: int) -> str | None:
    """Line naming under-filled categories, or None once everything is roughly even."""
    if not counts and target_total <= 0:
        return None
    fair_share = max(1, target_total // len(CATEGORIES))
    under = [c for c in CATEGORIES if counts.get(c, 0) < fair_share]
    if not under:
        return None
    return f"Prioritize these under-covered categories: {', '.join(under)}."


def generate_goldens(
    root: str | Path,
    *,
    count: int = 20,
    seed: int | None = None,
    model: str | None = None,
    balance: bool = False,
    on_batch: Callable[[int, int], None] | None = None,
) -> list[Golden]:
    """Generate ~`count` validated goldens for the repo at `root`.

    `balance` steers each batch's prompt toward categories under-represented
    so far (a hint, not a hard quota). `on_batch(batch_index, running_total)`
    fires after each batch's accepted items are folded in.
    """
    root_path = Path(root).expanduser().resolve()
    rng = random.Random(seed)
    goldens: list[Golden] = []
    seen_questions: set[str] = set()
    category_counts: Counter = Counter()
    # Each batch shows FILES_PER_BATCH files and asks for questions across them.
    # Batches can come back short (validation drops items) or fail outright
    # (provider error), so top up by attempts rather than a fixed batch count.
    batches = max(1, (count + 4) // 5)
    attempts = 0
    while len(goldens) < count and attempts < max(batches, count):
        attempts += 1
        files = _sample_files(root_path, FILES_PER_BATCH, rng)
        if not files:
            break
        sections = []
        for f in files:
            rel = f.relative_to(root_path).as_posix()
            sections.append(f"=== FILE: {rel} ===\n{_read_clipped(f)}")
        prompt = (
            "Repo files below. Generate 5-7 golden questions grounded in them.\n\n"
            + "\n\n".join(sections)
        )
        if balance:
            hint = _balance_hint(category_counts, count)
            if hint:
                prompt += "\n\n" + hint
        try:
            response = complete(
                GENERATION_SYSTEM,
                [{"role": "user", "content": prompt}],
                model=model,
            )
            items = extract_json(response.text)
        except LLMError as err:
            print(f"  batch skipped (provider error: {err})", file=sys.stderr)
            continue
        if not isinstance(items, list):
            continue
        for golden in _validate(items, root_path):
            if golden.question in seen_questions:
                continue
            seen_questions.add(golden.question)
            goldens.append(golden)
            category_counts[golden.category] += 1
        if on_batch:
            on_batch(attempts, len(goldens))
    return goldens[:count]


def save_goldens(goldens: list[Golden], path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(g) for g in goldens], indent=2) + "\n")
    return out


def load_goldens(path: str | Path) -> list[Golden]:
    data = json.loads(Path(path).expanduser().read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of goldens")
    return [Golden(**item) for item in data]
