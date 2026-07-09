"""Turn eval knowledge gaps into an agent onboarding doc (the context pack).

Reads the latest eval results, collects everything the agent got wrong or
partially wrong, and drafts `AGENTS.md.htc-draft`: for each gap, the fact plus
the artifact pointer an agent needs. Never overwrites an existing `AGENTS.md` —
the human reviews and merges, then re-runs `htc eval --compare` for the delta.
"""

from __future__ import annotations

from pathlib import Path

from ..evaluation.runner import EvalResult
from ..llm import complete

DRAFT_NAME = "AGENTS.md.htc-draft"

WRITER_SYSTEM = """You write the onboarding document that makes an AI agent \
company-native for a specific repository. You get knowledge gaps: questions a \
fresh agent answered wrong, each with the correct answer and the file that \
grounds it.

Write a markdown document that teaches these facts directly:
- Organize by theme (architecture, config, behavior, ops) with ## headings.
- State each fact plainly as operating knowledge, citing the file path inline. \
Write "X does Y (see path/to/file)" — never "the agent failed to know X".
- Merge related gaps into one coherent explanation instead of listing them.
- Be concise: an agent reads this at session start; every line must earn its place.
- Do not invent facts beyond the provided answers; do not add generic advice.

Output ONLY the markdown document body, starting with a single # title."""


def _gaps_prompt(result: EvalResult) -> str | None:
    gaps = [item for item in result.items if item.verdict != "correct"]
    if not gaps:
        return None
    lines = []
    for item in gaps:
        lines.append(
            f"- category: {item.golden.category}\n"
            f"  question: {item.golden.question}\n"
            f"  correct answer: {item.golden.answer}\n"
            f"  grounding file: {item.golden.artifact}"
        )
    return "Knowledge gaps from the latest eval (agent verdicts: wrong/partial):\n\n" + "\n".join(
        lines
    )


def write_context_pack(
    root: str | Path,
    result: EvalResult,
    *,
    model: str | None = None,
) -> Path | None:
    """Draft the context pack from eval gaps. Returns the draft path, or None
    when there are no gaps to write (score was perfect)."""
    root_path = Path(root).expanduser().resolve()
    prompt = _gaps_prompt(result)
    if prompt is None:
        return None
    response = complete(WRITER_SYSTEM, [{"role": "user", "content": prompt}], model=model)
    body = response.text.strip()
    if not body:
        raise RuntimeError("context-pack model reply was empty")
    draft = root_path / DRAFT_NAME
    draft.write_text(body + "\n")
    return draft
