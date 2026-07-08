"""
conftest.py — pytest configuration shared across the tests/ directory.

WHY this file exists: src/ isn't installed as a package (there's no
setup.py/pyproject.toml here — this is a small learning project, not a
distributable library), so `import ingest` would fail from tests/ without
src/ being on sys.path. Adding it once here, in a file pytest loads
automatically before collecting tests, means individual test files can use
plain `import ingest` / `import embed` instead of each repeating their own
sys.path hack.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
