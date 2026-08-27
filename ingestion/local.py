from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".py", ".ipynb", ".pdf", ".csv"}


@dataclass(frozen=True)
class Submission:
    student_name: str
    path: Path


def discover_submissions(root: str | Path) -> list[Submission]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    submissions = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS for p in path.rglob("*")):
            submissions.append(Submission(path.name, path))
    return submissions


def list_submission_files(submission: Submission) -> list[Path]:
    return sorted(
        p for p in submission.path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
