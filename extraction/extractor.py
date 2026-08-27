import json
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd
from pypdf import PdfReader


def extract_python(path: Path) -> dict[str, Any]:
    return {"type": "python", "text": path.read_text(encoding="utf-8", errors="replace")}


def extract_notebook(path: Path) -> dict[str, Any]:
    nb = nbformat.read(path, as_version=4)
    markdown, code, outputs = [], [], []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            markdown.append(cell.source)
        elif cell.cell_type == "code":
            code.append(cell.source)
            for output in cell.get("outputs", []):
                if "text" in output:
                    outputs.append(str(output["text"]))
                elif "data" in output and "text/plain" in output["data"]:
                    outputs.append(str(output["data"]["text/plain"]))
    return {
        "type": "notebook",
        "text": "\n\n".join(markdown + code),
        "markdown": "\n\n".join(markdown),
        "code": "\n\n".join(code),
        "outputs": "\n\n".join(outputs),
    }


def extract_pdf(path: Path) -> dict[str, Any]:
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return {"type": "pdf", "text": "\n\n".join(pages), "pages": len(pages)}


def extract_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    return {
        "type": "csv",
        "text": df.to_csv(index=False),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.astype(str).tolist(),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
    }


def extract_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return extract_python(path)
    if suffix == ".ipynb":
        return extract_notebook(path)
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".csv":
        return extract_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def extract_submission(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    files = []
    combined_text = []
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in {".py", ".ipynb", ".pdf", ".csv"}:
            try:
                extracted = extract_file(file_path)
                files.append({"path": str(file_path.relative_to(root)), **extracted})
                if extracted.get("text"):
                    combined_text.append(f"\n--- {file_path.name} ({extracted['type']}) ---\n{extracted['text']}")
            except Exception as exc:
                files.append({"path": str(file_path.relative_to(root)), "type": "error", "text": "", "error": str(exc)})
    return {"root": str(root), "files": files, "text": "\n".join(combined_text)}
