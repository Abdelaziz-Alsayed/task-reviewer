import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.pipeline import EvaluationResult


SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    submission_path TEXT NOT NULL,
    submitted_at TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    assignment_id TEXT NOT NULL,
    status TEXT NOT NULL,
    ai_score REAL,
    final_score REAL,
    max_score REAL NOT NULL,
    feedback TEXT,
    evaluated_at TEXT NOT NULL,
    FOREIGN KEY(student_id) REFERENCES students(id)
);

CREATE TABLE IF NOT EXISTS criterion_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL,
    criterion TEXT NOT NULL,
    max_score REAL NOT NULL,
    ai_score REAL,
    final_score REAL,
    evidence TEXT,
    feedback TEXT,
    missing_requirements TEXT,
    reasoning TEXT,
    confidence REAL,
    FOREIGN KEY(evaluation_id) REFERENCES evaluations(id)
);
"""


def connect(path: str | Path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str | Path):
    with connect(path) as conn:
        conn.executescript(SCHEMA)

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(criterion_results)"
            ).fetchall()
        }

        migrations = {
            "missing_requirements": (
                "ALTER TABLE criterion_results "
                "ADD COLUMN missing_requirements TEXT"
            ),
            "reasoning": (
                "ALTER TABLE criterion_results "
                "ADD COLUMN reasoning TEXT"
            ),
            "confidence": (
                "ALTER TABLE criterion_results "
                "ADD COLUMN confidence REAL"
            ),
        }

        for column, sql in migrations.items():
            if column not in columns:
                conn.execute(sql)

        conn.commit()


def save_evaluation_result(
    path: str | Path,
    result,
    assignment_id: str,
    final_score: float | None = None,
    feedback: str = "",
) -> int:
    """
    Persist a complete Phase 2 EvaluationResult into SQLite.

    The AI suggested score is stored separately from the optional
    instructor final score.
    """

    now = datetime.now(timezone.utc).isoformat()

    with connect(path) as conn:
        # ---------------------------------------------------------
        # 1. Create or update student
        # ---------------------------------------------------------
        conn.execute(
            """
            INSERT INTO students(
                name,
                submission_path,
                submitted_at
            )
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                submission_path = excluded.submission_path
            """,
            (
                result.student_name,
                result.submission_path,
                now,
            ),
        )

        student = conn.execute(
            """
            SELECT id
            FROM students
            WHERE name = ?
            """,
            (result.student_name,),
        ).fetchone()

        student_id = student["id"]

        # ---------------------------------------------------------
        # 2. Find existing evaluation
        # ---------------------------------------------------------
        existing = conn.execute(
            """
            SELECT id
            FROM evaluations
            WHERE student_id = ?
              AND assignment_id = ?
            """,
            (
                student_id,
                assignment_id,
            ),
        ).fetchone()

        status = (
            "reviewed"
            if final_score is not None
            else "evaluated"
        )

        ai_score = result.ai_suggested_score.total_score
        max_score = result.ai_suggested_score.total_marks

        if existing:
            evaluation_id = existing["id"]

            conn.execute(
                """
                DELETE FROM criterion_results
                WHERE evaluation_id = ?
                """,
                (evaluation_id,),
            )

            conn.execute(
                """
                UPDATE evaluations
                SET
                    status = ?,
                    ai_score = ?,
                    final_score = ?,
                    max_score = ?,
                    feedback = ?,
                    evaluated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    ai_score,
                    final_score,
                    max_score,
                    feedback,
                    now,
                    evaluation_id,
                ),
            )

        else:
            cursor = conn.execute(
                """
                INSERT INTO evaluations(
                    student_id,
                    assignment_id,
                    status,
                    ai_score,
                    final_score,
                    max_score,
                    feedback,
                    evaluated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    assignment_id,
                    status,
                    ai_score,
                    final_score,
                    max_score,
                    feedback,
                    now,
                ),
            )

            evaluation_id = cursor.lastrowid

        # ---------------------------------------------------------
        # 3. Save criterion-level results
        # ---------------------------------------------------------
        for criterion in result.ai_evaluation.criterion_results:
            conn.execute(
                """
                INSERT INTO criterion_results(
                    evaluation_id,
                    criterion,
                    max_score,
                    ai_score,
                    final_score,
                    evidence,
                    feedback,
                    missing_requirements,
                    reasoning,
                    confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    criterion.criterion_id,
                    criterion.max_score,
                    criterion.score,
                    None,
                    json.dumps(
                        criterion.evidence,
                        ensure_ascii=False,
                    ),
                    "",
                    json.dumps(
                        criterion.missing_requirements,
                        ensure_ascii=False,
                    ),
                    criterion.reasoning,
                    criterion.confidence,
                ),
            )

        conn.commit()

        return int(evaluation_id)

def save_human_review(
    path: str | Path,
    evaluation_id: int,
    criterion_scores: dict[str, float],
    final_score: float,
    feedback: str = "",
) -> None:
    now = datetime.now(timezone.utc).isoformat()

    with connect(path) as conn:
        evaluation = conn.execute(
            """
            SELECT max_score
            FROM evaluations
            WHERE id = ?
            """,
            (evaluation_id,),
        ).fetchone()

        if evaluation is None:
            raise ValueError(
                f"Evaluation {evaluation_id} does not exist."
            )

        max_score = float(evaluation["max_score"])

        if final_score < 0 or final_score > max_score:
            raise ValueError(
                f"Final score must be between 0 and {max_score}."
            )

        criteria = conn.execute(
            """
            SELECT id, criterion, max_score
            FROM criterion_results
            WHERE evaluation_id = ?
            """,
            (evaluation_id,),
        ).fetchall()

        criterion_map = {
            row["criterion"]: row
            for row in criteria
        }

        expected_ids = set(criterion_map)
        received_ids = set(criterion_scores)

        if expected_ids != received_ids:
            missing = expected_ids - received_ids
            unknown = received_ids - expected_ids

            message = []

            if missing:
                message.append(
                    "Missing criterion scores: "
                    + ", ".join(sorted(missing))
                )

            if unknown:
                message.append(
                    "Unknown criterion scores: "
                    + ", ".join(sorted(unknown))
                )

            raise ValueError("; ".join(message))

        for criterion_id, score in criterion_scores.items():
            score = float(score)
            criterion_max = float(
                criterion_map[criterion_id]["max_score"]
            )

            if score < 0 or score > criterion_max:
                raise ValueError(
                    f"Score for {criterion_id} must be "
                    f"between 0 and {criterion_max}."
                )

            conn.execute(
                """
                UPDATE criterion_results
                SET final_score = ?
                WHERE evaluation_id = ?
                  AND criterion = ?
                """,
                (
                    score,
                    evaluation_id,
                    criterion_id,
                ),
            )

        conn.execute(
            """
            UPDATE evaluations
            SET status = 'reviewed',
                final_score = ?,
                feedback = ?,
                evaluated_at = ?
            WHERE id = ?
            """,
            (
                final_score,
                feedback,
                now,
                evaluation_id,
            ),
        )

        conn.commit()

def list_evaluations(path: str | Path):
    with connect(path) as conn:
        return conn.execute(
            """
            SELECT
                s.name,
                e.assignment_id,
                e.status,
                e.ai_score,
                e.final_score,
                e.max_score,
                e.evaluated_at,
                e.id
            FROM evaluations e
            JOIN students s
                ON s.id = e.student_id
            ORDER BY e.evaluated_at DESC
            """
        ).fetchall()

def save_evaluation_result(
    path: str | Path,
    result,
    assignment_id: str,
    final_score: float | None = None,
    feedback: str = "",
) -> int:
    """
    Persist a complete Phase 2 EvaluationResult into SQLite.

    The AI suggested score is stored separately from the optional
    instructor final score.
    """

    now = datetime.now(timezone.utc).isoformat()

    with connect(path) as conn:
        conn.execute(
            """
            INSERT INTO students(
                name,
                submission_path,
                submitted_at
            )
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                submission_path = excluded.submission_path
            """,
            (
                result.student_name,
                result.submission_path,
                now,
            ),
        )

        student = conn.execute(
            """
            SELECT id
            FROM students
            WHERE name = ?
            """,
            (result.student_name,),
        ).fetchone()

        student_id = student["id"]

        existing = conn.execute(
            """
            SELECT id
            FROM evaluations
            WHERE student_id = ?
              AND assignment_id = ?
            """,
            (
                student_id,
                assignment_id,
            ),
        ).fetchone()

        status = (
            "reviewed"
            if final_score is not None
            else "evaluated"
        )

        ai_score = result.ai_suggested_score.total_score
        max_score = result.ai_suggested_score.total_marks

        if existing:
            evaluation_id = existing["id"]

            conn.execute(
                """
                DELETE FROM criterion_results
                WHERE evaluation_id = ?
                """,
                (evaluation_id,),
            )

            conn.execute(
                """
                UPDATE evaluations
                SET
                    status = ?,
                    ai_score = ?,
                    final_score = ?,
                    max_score = ?,
                    feedback = ?,
                    evaluated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    ai_score,
                    final_score,
                    max_score,
                    feedback,
                    now,
                    evaluation_id,
                ),
            )

        else:
            cursor = conn.execute(
                """
                INSERT INTO evaluations(
                    student_id,
                    assignment_id,
                    status,
                    ai_score,
                    final_score,
                    max_score,
                    feedback,
                    evaluated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    assignment_id,
                    status,
                    ai_score,
                    final_score,
                    max_score,
                    feedback,
                    now,
                ),
            )

            evaluation_id = cursor.lastrowid

        for criterion in result.ai_evaluation.criterion_results:
            conn.execute(
                """
                INSERT INTO criterion_results(
                    evaluation_id,
                    criterion,
                    max_score,
                    ai_score,
                    final_score,
                    evidence,
                    feedback,
                    missing_requirements,
                    reasoning,
                    confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    criterion.criterion_id,
                    criterion.max_score,
                    criterion.score,
                    None,
                    json.dumps(
                        criterion.evidence,
                        ensure_ascii=False,
                    ),
                    "",
                    json.dumps(
                        criterion.missing_requirements,
                        ensure_ascii=False,
                    ),
                    criterion.reasoning,
                    criterion.confidence,
                ),
            )

        conn.commit()

        return int(evaluation_id)
    
def get_evaluation(
    path: str | Path,
    evaluation_id: int,
):
    with connect(path) as conn:
        evaluation = conn.execute(
            """
            SELECT
                e.*,
                s.name,
                s.submission_path
            FROM evaluations e
            JOIN students s
                ON s.id = e.student_id
            WHERE e.id = ?
            """,
            (evaluation_id,),
        ).fetchone()

        criteria = conn.execute(
            """
            SELECT *
            FROM criterion_results
            WHERE evaluation_id = ?
            ORDER BY id
            """,
            (evaluation_id,),
        ).fetchall()

        return evaluation, criteria