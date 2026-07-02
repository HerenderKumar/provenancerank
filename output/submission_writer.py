"""Assemble and write submission.csv, then run the official validator on it.

The fiddly bit is the tie-break rule. The validator wants scores non-increasing
by rank AND, for equal scores, candidate_id ascending. Float scores rarely tie
until you round them to 4 dp for the file — at which point neighbours can become
equal. So: round first, THEN sort by (score desc, candidate_id asc), then assign
ranks. That ordering satisfies both checks no matter how the rounding lands.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pandas as pd

from core.logging import get_logger

log = get_logger("output.submission_writer")

SCORE_DP = 4
HEADER = ["candidate_id", "rank", "score", "reasoning"]


def build_submission(
    df: pd.DataFrame,
    top_k: int = 100,
    score_col: str = "final_score",
    reasoning_col: str = "reasoning",
) -> pd.DataFrame:
    """df must contain candidate_id, the score column, and reasoning. Returns a
    clean top-k frame with rank/score/reasoning, ordered and tie-broken."""
    work = df[["candidate_id", score_col, reasoning_col]].copy()
    work["score"] = work[score_col].astype(float).round(SCORE_DP)
    # round, then order by score desc and id asc — this is what makes the
    # validator's tie-break check pass.
    work = work.sort_values(["score", "candidate_id"], ascending=[False, True])
    work = work.head(top_k).reset_index(drop=True)
    work["rank"] = range(1, len(work) + 1)
    out = work[["candidate_id", "rank", "score", reasoning_col]].rename(
        columns={reasoning_col: "reasoning"}
    )
    return out


def write_submission(submission: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(HEADER)
        for _, r in submission.iterrows():
            score = f"{float(r['score']):.{SCORE_DP}f}"
            reasoning = " ".join(str(r["reasoning"]).split())  # no stray newlines
            w.writerow([r["candidate_id"], int(r["rank"]), score, reasoning])
    log.info("submission.written", path=str(path), rows=len(submission))
    return path


def _load_official_validator():
    """Import the unmodified validate_submission.py shipped by Redrob."""
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "redrob_validator", root / "validate_submission.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def validate(path: str | Path) -> list[str]:
    """Return the official validator's error list (empty == valid)."""
    validator = _load_official_validator()
    return validator.validate_submission(str(path))


def write_and_validate(submission: pd.DataFrame, path: str | Path) -> Path:
    path = write_submission(submission, path)
    errors = validate(path)
    if errors:
        for e in errors:
            log.error("submission.invalid", detail=e)
        raise ValueError(f"submission failed validation ({len(errors)} issues): {errors}")
    log.info("submission.valid", path=str(path))
    return path
