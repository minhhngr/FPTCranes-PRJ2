"""Repository-local entry point for offline training validation.

This wrapper makes the CLI runnable from a source checkout without installing the
``src`` package or manually setting ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __name__ == "__main__":
    main = import_module("ai_job_market.training_validation").main
    raise SystemExit(main())
