from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = ROOT_DIR / "data" / "submissions"
EXPORTS_DIR = ROOT_DIR / "data" / "exports"
DB_PATH = ROOT_DIR / "evaluation.db"
RUBRIC_PATH = ROOT_DIR / "config" / "rubric.yaml"
