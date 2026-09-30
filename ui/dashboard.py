import streamlit as st
import pandas as pd
import json
import time

from config.rubric import load_rubric
from config.settings import DB_PATH, RUBRIC_PATH, SUBMISSIONS_DIR
from database.db import get_evaluation, init_db, list_evaluations
from database.db import save_human_review
from evaluation.history import (
    get_evaluation_details,
    get_evaluation_history,
)
from export.reports import export_summary_csv
from ingestion.local import discover_submissions, list_submission_files
from evaluation.runner import run_evaluation
from llm.factory import create_llm_evaluator


def render_dashboard(rubric):
    rows = [dict(r) for r in list_evaluations(DB_PATH)]
    total = len(rows)
    evaluated = sum(r["status"] in {"evaluated", "reviewed"} for r in rows)
    pending = max(len(discover_submissions(SUBMISSIONS_DIR)) - evaluated, 0)
    scores = [r["final_score"] if r["final_score"] is not None else r["ai_score"] for r in rows if (r["final_score"] is not None or r["ai_score"] is not None)]
    st.title("Task Reviewer")
    st.caption(rubric["assignment"]["name"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Submissions", len(discover_submissions(SUBMISSIONS_DIR)))
    c2.metric("Evaluated", evaluated)
    c3.metric("Pending", pending)
    c4.metric("Average", f"{sum(scores)/len(scores):.1f} / {rubric['assignment']['total_marks']}" if scores else "—")
    st.divider()
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df[["name", "status", "ai_score", "final_score", "max_score"]], use_container_width=True, hide_index=True)
    else:
        st.info("No evaluations yet. Use the Submissions page to evaluate local submissions.")


def render_submissions(rubric):
    st.header("Submissions")
    provider = st.radio(
        "LLM Provider",
        ["Mock", "Gemini"],
        horizontal=True,
        help=(
            "Mock runs locally without using an LLM API. "
            "Gemini performs a real AI evaluation."
        ),
    )
    provider_key = provider.lower()

    submissions = discover_submissions(SUBMISSIONS_DIR)
    evaluations = {
                    row["name"]: dict(row)
                    for row in list_evaluations(DB_PATH)
                }
    if not submissions:
        st.info(f"Add one student folder under `{SUBMISSIONS_DIR}`.")
        return
    
    evaluations = {
        row["name"]: dict(row)
        for row in list_evaluations(DB_PATH)
    }

    for submission in submissions:
        with st.container(border=True):
            st.subheader(submission.student_name)

            files = list_submission_files(submission)

            st.write(
                f"{len(files)} supported file(s)"
            )

            st.code(
                "\n".join(
                    str(p.relative_to(submission.path))
                    for p in files
                ),
                language="text",
            )

            existing_evaluation = evaluations.get(
                submission.student_name
            )

            if existing_evaluation:
                st.info(
                    "Already evaluated — "
                    f"{existing_evaluation['status'].capitalize()}"
                )

                button_label = "Re-evaluate"
                button_key = (
                    f"reeval-{submission.student_name}"
                )
            else:
                button_label = "Evaluate"
                button_key = (
                    f"eval-{submission.student_name}"
                )

            if st.button(
                button_label,
                key=button_key,
            ):
                llm_evaluator = create_llm_evaluator(
                    provider_key
                )

                action = (
                    "Re-evaluating"
                    if existing_evaluation
                    else "Evaluating"
                )

                try:
                    with st.spinner(
                        f"{action} "
                        f"{submission.student_name}..."
                    ):
                        result = run_evaluation(
                            submission_path=submission.path,
                            student_name=submission.student_name,
                            assignment=rubric["assignment"],
                            rubric=rubric["criteria"],
                            llm_evaluator=llm_evaluator,
                            db_path=DB_PATH,
                        )

                except Exception as exc:
                    st.error(
                        "The evaluation could not be completed."
                    )

                    st.caption(
                        f"Error details: {exc}"
                    )

                else:
                    st.success(
                        "AI evaluation completed: "
                        f"{result.ai_suggested_score.total_score:.1f} / "
                        f"{result.ai_suggested_score.total_marks:.1f}"
                    )

                    time.sleep(3)
                    st.rerun()


def render_evaluation():
    st.header("Evaluation Review")

    rows = [dict(row) for row in list_evaluations(DB_PATH)]

    if not rows:
        st.info("No evaluations available.")
        return

    labels = {
        row["id"]: (
            f"{row['name']} — "
            f"{row['ai_score']:.1f}/{row['max_score']}"
        )
        for row in rows
    }

    selected = st.selectbox(
        "Student",
        list(labels),
        format_func=lambda evaluation_id: labels[evaluation_id],
    )

    evaluation, criteria = get_evaluation(
        DB_PATH,
        selected,
    )

    if evaluation is None:
        st.error("Evaluation not found.")
        return

    st.subheader(evaluation["name"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "AI Suggested Score",
        f"{evaluation['ai_score']:.1f} / {evaluation['max_score']}",
    )

    if evaluation["final_score"] is not None:
        col2.metric(
            "Final Score",
            f"{evaluation['final_score']:.1f} / "
            f"{evaluation['max_score']}",
        )
    else:
        col2.metric(
            "Final Score",
            "Not reviewed",
        )

    col3.metric(
        "Status",
        evaluation["status"].capitalize(),
    )

    st.divider()

    st.subheader("Criterion Review")

    criterion_scores = {}

    for criterion in criteria:
        criterion_id = criterion["criterion"]
        max_score = float(criterion["max_score"])
        ai_score = float(criterion["ai_score"])

        st.markdown(
            f"### {criterion_id}"
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.write(
                f"**AI Score:** "
                f"{ai_score:.1f} / {max_score}"
            )

            if criterion["confidence"] is not None:
                st.write(
                    f"**Confidence:** "
                    f"{float(criterion['confidence']):.0%}"
                )

        with col2:
            existing_final = criterion["final_score"]

            default_score = (
                float(existing_final)
                if existing_final is not None
                else ai_score
            )

            final_score = st.number_input(
                "Final criterion score",
                min_value=0.0,
                max_value=max_score,
                value=default_score,
                step=0.5,
                key=f"final-score-{selected}-{criterion_id}",
            )

            criterion_scores[criterion_id] = final_score

        if criterion["evidence"]:
            st.markdown("**AI Evidence**")

            evidence = json.loads(
                criterion["evidence"]
            )

            for item in evidence:
                st.write(f"- {item}")

        if criterion["missing_requirements"]:
            st.markdown("**Missing Requirements**")

            missing = json.loads(
                criterion["missing_requirements"]
            )

            for item in missing:
                st.write(f"- {item}")

        if criterion["reasoning"]:
            st.markdown("**AI Reasoning**")
            st.write(criterion["reasoning"])

        st.divider()

    calculated_total = sum(
        criterion_scores.values()
    )

    st.subheader("Final Review")

    st.metric(
        "Calculated Final Score",
        f"{calculated_total:.1f} / "
        f"{evaluation['max_score']}",
    )

    feedback = st.text_area(
        "Instructor Feedback",
        value=evaluation["feedback"] or "",
        placeholder="Add feedback for the student...",
    )

    if st.button(
        "Save Human Review",
        type="primary",
    ):
        save_human_review(
            path=DB_PATH,
            evaluation_id=selected,
            criterion_scores=criterion_scores,
            final_score=calculated_total,
            feedback=feedback,
        )

        st.success(
            "Human review saved successfully."
        )

        time.sleep(3)

        st.rerun()

def render_history():
    st.header("Evaluation History")

    history = get_evaluation_history(DB_PATH)

    if not history:
        st.info("No evaluations available.")
        return

    df = pd.DataFrame(history)

    display_columns = [
        "id",
        "name",
        "assignment_id",
        "status",
        "ai_score",
        "final_score",
        "max_score",
        "evaluated_at",
    ]

    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    evaluation_ids = [row["id"] for row in history]

    labels = {
        row["id"]: (
            f"{row['name']} — "
            f"{row['ai_score']:.1f}/{row['max_score']}"
        )
        for row in history
    }

    selected = st.selectbox(
        "View evaluation",
        evaluation_ids,
        format_func=lambda evaluation_id: labels[evaluation_id],
    )

    evaluation, criteria = get_evaluation_details(
        DB_PATH,
        selected,
    )

    if evaluation is None:
        st.error("Evaluation not found.")
        return

    st.subheader(evaluation["name"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "AI Suggested Score",
        f"{evaluation['ai_score']:.1f} / {evaluation['max_score']}",
    )

    final_score = evaluation["final_score"]

    col2.metric(
        "Final Score",
        (
            f"{final_score:.1f} / {evaluation['max_score']}"
            if final_score is not None
            else "Not reviewed"
        ),
    )

    col3.metric(
        "Status",
        evaluation["status"].capitalize(),
    )

    st.caption(
        f"Assignment: {evaluation['assignment_id']}"
    )

    st.caption(
        f"Evaluated at: {evaluation['evaluated_at']}"
    )

    st.divider()

    st.subheader("Criterion Results")

    for criterion in criteria:
        st.markdown(
            f"### {criterion['criterion']}"
        )

        st.write(
            f"**AI Score:** "
            f"{criterion['ai_score']:.1f} / "
            f"{criterion['max_score']}"
        )

        if criterion["confidence"] is not None:
            st.write(
                f"**Confidence:** "
                f"{criterion['confidence']:.0%}"
            )

        if criterion["evidence"]:
            st.write("**Evidence**")

            import json

            evidence = json.loads(
                criterion["evidence"]
            )

            for item in evidence:
                st.write(f"- {item}")

        if criterion["missing_requirements"]:
            st.write("**Missing Requirements**")

            import json

            missing = json.loads(
                criterion["missing_requirements"]
            )

            for item in missing:
                st.write(f"- {item}")

        if criterion["reasoning"]:
            st.write("**Reasoning**")
            st.write(criterion["reasoning"])

        st.divider()

def render_export():
    st.header("Export")
    if st.button("Generate CSV"):
        path = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "exports" / "evaluation_summary.csv"
        df = export_summary_csv(DB_PATH, path)
        st.download_button("Download CSV", df.to_csv(index=False), "evaluation_summary.csv", "text/csv")


def run():
    rubric = load_rubric(RUBRIC_PATH)

    init_db(DB_PATH)

    page = st.sidebar.radio(
        "Page",
        [
            "Dashboard",
            "Submissions",
            "Evaluation",
            "History",
            "Export",
        ],
    )

    if page == "Dashboard":
        render_dashboard(rubric)

    elif page == "Submissions":
        render_submissions(rubric)

    elif page == "Evaluation":
        render_evaluation()

    elif page == "History":
        render_history()

    else:
        render_export()
