#!/usr/bin/env python
"""Generate or check supplemental diagnostic evidence for UI pages 04/05.

Cross-platform — works on Windows, macOS, and Linux with plain ``python``.

Usage
-----
    python scripts/run_ui_evidence.py                  # generate evidence
    python scripts/run_ui_evidence.py --check          # verify freshness only
    python scripts/run_ui_evidence.py --workspace .    # explicit workspace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure both ``src/`` and the project root are importable.
# The project root is needed because the serialised model bundle references
# ``src.ai_job_market.core.SkillMultiHotEncoder`` via pickle.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for directory in (str(SRC_DIR), str(PROJECT_ROOT)):
    if directory not in sys.path:
        sys.path.insert(0, directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or validate supplemental UI evidence for pages 04/05.",
    )
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT),
        help="Workspace root that contains outputs/ and artifacts/ (default: project root).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that existing evidence is valid; do not regenerate.",
    )
    args = parser.parse_args(argv)

    from ai_job_market.ui_evidence import check_workspace, generate

    workspace = Path(args.workspace).resolve()

    if args.check:
        result = check_workspace(workspace)
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 2

    try:
        result = generate(workspace)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"ui-evidence generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
