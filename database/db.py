import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

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


def save_evaluation(path: str | Path, student_name: str, submission_path: str, assignment_id: str, result: dict, final_score: float | None = None, feedback: str = ""):
    now = datetime.now(timezone.utc).isoformat()
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO students(name, submission_path, submitted_at) VALUES (?, ?, ?) ON CONFLICT(name) DO UPDATE SET submission_path=excluded.submission_path",
            (student_name, submission_path, now),
        )
        student_id = conn.execute("SELECT id FROM students WHERE name=?", (student_name,)).fetchone()["id"]
        existing = conn.execute("SELECT id FROM evaluations WHERE student_id=? AND assignment_id=?", (student_id, assignment_id)).fetchone()
        if existing:
            evaluation_id = existing["id"]
            conn.execute("DELETE FROM criterion_results WHERE evaluation_id=?", (evaluation_id,))
            conn.execute("UPDATE evaluations SET status=?, ai_score=?, final_score=?, max_score=?, feedback=?, evaluated_at=? WHERE id=?", ("reviewed" if final_score is not None else "evaluated", result["score"], final_score, result["max_score"], feedback, now, evaluation_id))
        else:
            cur = conn.execute("INSERT INTO evaluations(student_id, assignment_id, status, ai_score, final_score, max_score, feedback, evaluated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (student_id, assignment_id, "reviewed" if final_score is not None else "evaluated", result["score"], final_score, result["max_score"], feedback, now))
            evaluation_id = cur.lastrowid
        for criterion in result["criteria"]:
            conn.execute("INSERT INTO criterion_results(evaluation_id, criterion, max_score, ai_score, final_score, evidence, feedback) VALUES (?, ?, ?, ?, ?, ?, ?)", (evaluation_id, criterion["criterion"], criterion["max_score"], criterion["score"], None, json.dumps(criterion["evidence"]), ""))
        conn.commit()
        return evaluation_id


def list_evaluations(path: str | Path):
    with connect(path) as conn:
        return conn.execute("""
            SELECT s.name, e.status, e.ai_score, e.final_score, e.max_score, e.evaluated_at, e.id
            FROM evaluations e JOIN students s ON s.id=e.student_id
            ORDER BY s.name
        """).fetchall()


def get_evaluation(path: str | Path, evaluation_id: int):
    with connect(path) as conn:
        evaluation = conn.execute("""
            SELECT e.*, s.name, s.submission_path
            FROM evaluations e JOIN students s ON s.id=e.student_id
            WHERE e.id=?
        """, (evaluation_id,)).fetchone()
        criteria = conn.execute("SELECT * FROM criterion_results WHERE evaluation_id=? ORDER BY id", (evaluation_id,)).fetchall()
        return evaluation, criteria
