import pytest, tempfile
from pathlib import Path
from planagent.state import create_initial_state
from planagent.context_reader import read_context

def test_empty_folder():
    with tempfile.TemporaryDirectory() as tmp:
        s = create_initial_state()
        s = read_context(tmp, s)
        assert s["scenario"] == "empty"
        assert s["existing_summary"] is None

def test_python_project_detected():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "requirements.txt").write_text("fastapi\nsqlalchemy\n")
        Path(tmp, "app").mkdir()
        Path(tmp, "app", "main.py").write_text("from fastapi import FastAPI")
        s = create_initial_state()
        s = read_context(tmp, s)
        assert s["scenario"] == "existing"
        assert s["existing_summary"]["language"] == "Python"
        assert s["existing_summary"]["framework"] == "FastAPI"
