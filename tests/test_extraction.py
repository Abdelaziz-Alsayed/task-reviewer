from pathlib import Path
from extraction.extractor import extract_file


def test_python_extraction(tmp_path: Path):
    p = tmp_path / "x.py"
    p.write_text("print('hello')", encoding="utf-8")
    result = extract_file(p)
    assert result["type"] == "python"
    assert "hello" in result["text"]


def test_csv_extraction(tmp_path: Path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    result = extract_file(p)
    assert result["rows"] == 1
    assert result["columns"] == 2
