#!/usr/bin/env python
"""Run the full offline pipeline (training + artifacts).

Cross-platform — works on Windows, macOS, and Linux with plain ``python``.

Usage
-----
    python scripts/run_training.py                         # default dataset
    python scripts/run_training.py --data other.csv        # custom CSV
    python scripts/run_training.py --workspace runs/demo   # isolated workspace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure ``src/`` is importable regardless of the working directory or OS.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full offline pipeline (Common + Branch A + Branch B).",
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "raw" / "ai_jobs_market_2025_2026.csv"),
        help="Path to the raw CSV dataset (default: data/raw/ai_jobs_market_2025_2026.csv).",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Isolate outputs/artifacts into WORKSPACE instead of the project root.",
    )
    args = parser.parse_args(argv)

    raw_path = Path(args.data).resolve()
    if not raw_path.is_file():
        print(f"ERROR: dataset not found: {raw_path}", file=sys.stderr)
        return 1

    workspace_root = Path(args.workspace).resolve() if args.workspace else None

    from ai_job_market.core import run_pipeline

    print(f"Dataset   : {raw_path}")
    print(f"Workspace : {workspace_root or PROJECT_ROOT}")
    print()

    result = run_pipeline(
        raw_path=raw_path,
        root=PROJECT_ROOT,
        workspace_root=workspace_root,
    )
    print(f"\nPipeline finished: {result.get('status', 'done')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
