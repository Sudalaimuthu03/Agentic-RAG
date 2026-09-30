"""Release gate for the RAG V2 project.

Compiles project source only; it intentionally excludes venv/.venv and other
runtime/build directories.  Full runtime gates that require Ollama are run by
the developer on the target machine.
"""
from __future__ import annotations

import compileall
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [ROOT / "rag", ROOT / "tests", ROOT / "scripts"]
ROOT_FILES = [ROOT / "app.py"]


def main() -> int:
    if sys.version_info[:2] != (3, 11):
        print(f"RELEASE_GATE_FAIL: Python 3.11 required; found {sys.version.split()[0]}")
        return 1

    failures = []
    for path in ROOT_FILES:
        if path.exists() and not compileall.compile_file(str(path), quiet=1):
            failures.append(str(path.relative_to(ROOT)))
    for directory in SOURCE_DIRS:
        if directory.exists() and not compileall.compile_dir(str(directory), quiet=1, maxlevels=20, rx=lambda p: any(part in {"venv", ".venv", ".git", "__pycache__"} for part in Path(p).parts)):
            failures.append(str(directory.relative_to(ROOT)))

    if failures:
        print("RELEASE_GATE_FAIL: compilation failures:")
        for item in failures:
            print(f" - {item}")
        return 1

    if importlib.util.find_spec("app") is None:
        print("RELEASE_GATE_FAIL: app module could not be located")
        return 1

    print("RELEASE_GATE_PASS: Python 3.11 source compilation passed")
    print("RELEASE_GATE_NOTE: runtime import/startup, Ollama, pytest, and behavioral regression must be executed in the target venv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
