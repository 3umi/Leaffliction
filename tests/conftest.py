"""
Test-suite setup: force a headless backend before pyplot is imported anywhere.
"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

# Ensure the project root is importable so `import Distribution` works.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
