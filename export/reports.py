import pandas as pd
from pathlib import Path
from database.db import list_evaluations


def export_summary_csv(db_path, output_path):
    rows = [dict(r) for r in list_evaluations(db_path)]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["name", "status", "ai_score", "final_score", "max_score", "evaluated_at"]]
        df.columns = ["student", "status", "suggested_score", "final_score", "max_score", "evaluated_at"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
