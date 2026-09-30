from database.db import (
    connect,
    init_db,
    save_human_review,
)


def test_save_human_review(tmp_path):
    db_path = tmp_path / "review.db"

    init_db(db_path)

    with connect(db_path) as conn:
        student_id = conn.execute(
            """
            INSERT INTO students(name, submission_path, submitted_at)
            VALUES (?, ?, ?)
            """,
            (
                "Test Student",
                "/tmp/submission",
                "2026-01-01T00:00:00+00:00",
            ),
        ).lastrowid

        evaluation_id = conn.execute(
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
                "week05_regression",
                "evaluated",
                22.0,
                None,
                25.0,
                "",
                "2026-01-01T00:00:00+00:00",
            ),
        ).lastrowid

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
                "evaluation",
                4.0,
                3.0,
                None,
                "[]",
                "",
                "[]",
                "AI reasoning",
                0.8,
            ),
        )

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
                "randomized_search",
                4.0,
                4.0,
                None,
                "[]",
                "",
                "[]",
                "AI reasoning",
                0.95,
            ),
        )

        conn.commit()

    save_human_review(
        path=db_path,
        evaluation_id=evaluation_id,
        criterion_scores={
            "evaluation": 2.0,
            "randomized_search": 4.0,
        },
        final_score=6.0,
        feedback="Reviewed manually.",
    )

    with connect(db_path) as conn:
        evaluation = conn.execute(
            """
            SELECT status, final_score, feedback
            FROM evaluations
            WHERE id = ?
            """,
            (evaluation_id,),
        ).fetchone()

        criteria = conn.execute(
            """
            SELECT criterion, ai_score, final_score
            FROM criterion_results
            WHERE evaluation_id = ?
            ORDER BY criterion
            """,
            (evaluation_id,),
        ).fetchall()

    assert evaluation["status"] == "reviewed"
    assert evaluation["final_score"] == 6.0
    assert evaluation["feedback"] == "Reviewed manually."

    scores = {
        row["criterion"]: row["final_score"]
        for row in criteria
    }

    assert scores["evaluation"] == 2.0
    assert scores["randomized_search"] == 4.0