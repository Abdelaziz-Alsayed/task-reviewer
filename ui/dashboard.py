import streamlit as st
import pandas as pd

from config.settings import DB_PATH, SUBMISSIONS_DIR
from database.db import get_evaluation, init_db, list_evaluations
from export.reports import export_summary_csv
from ingestion.local import discover_submissions, list_submission_files
from extraction.extractor import extract_submission
from evaluation.scoring import evaluate
from config.rubric import load_rubric


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
    submissions = discover_submissions(SUBMISSIONS_DIR)
    if not submissions:
        st.info(f"Add one student folder under `{SUBMISSIONS_DIR}`.")
        return
    for submission in submissions:
        with st.container(border=True):
            st.subheader(submission.student_name)
            files = list_submission_files(submission)
            st.write(f"{len(files)} supported file(s)")
            st.code("\n".join(str(p.relative_to(submission.path)) for p in files), language="text")
            if st.button("Evaluate", key=f"eval-{submission.student_name}"):
                extracted = extract_submission(submission.path)
                result = evaluate(extracted, rubric)
                from database.db import save_evaluation
                save_evaluation(DB_PATH, submission.student_name, str(submission.path), rubric["assignment"]["id"], result)
                st.success(f"Deterministic evaluation saved: {result['score']} / {result['max_score']}")
                st.rerun()


def render_evaluation():
    st.header("Evaluation Review")
    rows = [dict(r) for r in list_evaluations(DB_PATH)]
    if not rows:
        st.info("No evaluations available.")
        return
    labels = {r["id"]: f"{r['name']} — {r['ai_score']:.1f}/{r['max_score']}" for r in rows}
    selected = st.selectbox("Student", list(labels), format_func=lambda x: labels[x])
    evaluation, criteria = get_evaluation(DB_PATH, selected)
    st.subheader(evaluation["name"])
    st.metric("Deterministic suggested score", f"{evaluation['ai_score']:.1f} / {evaluation['max_score']}")
    for criterion in criteria:
        st.markdown(f"**{criterion['criterion']} — {criterion['ai_score']:.1f} / {criterion['max_score']}**")
        import json
        for evidence in json.loads(criterion["evidence"]):
            icon = "✓" if evidence["passed"] else "✗"
            st.write(f"{icon} {evidence['evidence']}")
    st.divider()
    final = st.number_input("Final score", min_value=0.0, max_value=float(evaluation["max_score"]), value=float(evaluation["final_score"] if evaluation["final_score"] is not None else evaluation["ai_score"]), step=0.5)
    if st.button("Save final score"):
        with __import__("sqlite3").connect(DB_PATH) as conn:
            conn.execute("UPDATE evaluations SET final_score=?, status='reviewed' WHERE id=?", (final, selected))
            conn.commit()
        st.success("Final score saved.")
        st.rerun()


def render_export():
    st.header("Export")
    if st.button("Generate CSV"):
        path = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "exports" / "evaluation_summary.csv"
        df = export_summary_csv(DB_PATH, path)
        st.download_button("Download CSV", df.to_csv(index=False), "evaluation_summary.csv", "text/csv")


def run():
    rubric = load_rubric(__import__("config.settings", fromlist=["RUBRIC_PATH"]).RUBRIC_PATH)
    init_db(DB_PATH)
    page = st.sidebar.radio("Page", ["Dashboard", "Submissions", "Evaluation", "Export"])
    if page == "Dashboard":
        render_dashboard(rubric)
    elif page == "Submissions":
        render_submissions(rubric)
    elif page == "Evaluation":
        render_evaluation()
    else:
        render_export()
