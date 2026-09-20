# Quickstart and Verification: Readable Training Logs

**Not an execution record.** These commands are for after task approval and governance disposition. The new flag and callable arguments are planned, not currently available. Never run `pipeline.py` in the original project root during validation: it has no workspace flag and would replace baseline outputs.

## 1. Use the existing environment

```bash
export PROJECT=/home/mihuynh-ubuntu/minhhuynh/FPTCranes/FPTCranes-PRJ2
export PY="$PROJECT/.venv/bin/python"
cd "$PROJECT"
"$PY" --version
```

Do not install/upgrade dependencies for logging work. First capture the current uncommitted working-tree baseline as described below; a checkout of HEAD omits inherited audit changes.

## 2. Capture the pre-change baseline (before implementation)

```bash
export BASELINE=$(mktemp -d /tmp/readable-training-baseline-XXXXXX)
"$PY" - <<'PY'
import os, shutil
from pathlib import Path
src, dst = Path(os.environ['PROJECT']), Path(os.environ['BASELINE'])
for name in ('src', 'config', 'data'):
    shutil.copytree(src / name, dst / name, ignore=shutil.ignore_patterns('__pycache__'))
for name in ('pipeline.py', 'requirements.txt', 'pyproject.toml'):
    shutil.copy2(src / name, dst / name)
PY
(cd "$BASELINE" && PYTHONPATH="$BASELINE/src" "$PY" "$BASELINE/pipeline.py" --no-heartbeat > "$BASELINE/terminal.txt" 2>&1)
```

Record the exit code, source/config hashes, runtime and baseline path in `specs/002-readable-training-logs/verification.md`. This is a full existing pipeline run, not a cheap smoke test; do not claim success until it actually completes. If governance blocks training, stop before this command.

## 3. Tests first, then focused verification

Write failing tests before implementing each behavior; record expected failures. Preserve current scientific characterization and move structured assertions from stdout to `.logs` file reads.

Planned focused suite, after the new test files exist:

```bash
cd "$PROJECT"
"$PY" -m pytest -q \
  tests/test_training_behavior.py \
  tests/test_training_audit.py \
  tests/test_training_audit_integration.py \
  tests/test_training_fold_audit.py \
  tests/test_training_audit_docs.py \
  tests/test_training_console.py \
  tests/test_training_stage_coverage.py \
  tests/test_training_evaluation_logs.py \
  tests/test_training_evidence_exports.py \
  tests/test_training_cli.py
"$PY" -m pytest -q

git diff --check
```

New tests cover no-color/redirected/narrow output, normal/debug parity, 0/20/21/large row tables, non-finite metrics, missing train scores, repeated IDs, same-month folds, logging initialization/write/close failure, observer failure, missing stage events, failed trial, non-RF tuning skip and interruption. Small/empty datasets must preserve current error semantics, not invent successful empty training.

## 4. Real normal/debug CLI in separate clean copies

After implementation, prepare both copies from the same modified working tree:

```bash
export NORMAL=$(mktemp -d /tmp/readable-training-normal-XXXXXX)
export DEBUG=$(mktemp -d /tmp/readable-training-debug-XXXXXX)
"$PY" - <<'PY'
import os, shutil
from pathlib import Path
src = Path(os.environ['PROJECT'])
for key in ('NORMAL', 'DEBUG'):
    dst = Path(os.environ[key])
    for name in ('src', 'config', 'data'):
        shutil.copytree(src / name, dst / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('pipeline.py', 'requirements.txt', 'pyproject.toml'):
        shutil.copy2(src / name, dst / name)
PY
(cd "$NORMAL" && PYTHONPATH="$NORMAL/src" "$PY" "$NORMAL/pipeline.py" --no-heartbeat > "$NORMAL/terminal.txt" 2>&1)
(cd "$DEBUG" && PYTHONPATH="$DEBUG/src" "$PY" "$DEBUG/pipeline.py" --debuglog --no-heartbeat > "$DEBUG/terminal.txt" 2>&1)
```

No links back to original output/artifact folders. The copies intentionally exclude old outputs and artifacts so stale evidence cannot make tests pass. Record every exit code; a pipeline launch or a growing log is not completed validation. Unit tests separately exercise heartbeat output with a fake clock, avoiding extra full training runs just for animation.

## 5. Direct core invocation in an isolated workspace

```bash
export CORE_WORKSPACE=$(mktemp -d /tmp/readable-training-core-XXXXXX)
PYTHONPATH="$PROJECT/src" "$PY" - <<'PY'
import os
from pathlib import Path
from ai_job_market.core import run_pipeline
project = Path(os.environ['PROJECT'])
summary = run_pipeline(
    raw_path=project / 'data/raw/ai_jobs_market_2025_2026.csv',
    root=project,
    workspace_root=Path(os.environ['CORE_WORKSPACE']),
    debuglog=True,
)
assert summary['workspace_root'] == os.environ['CORE_WORKSPACE']
PY
```

This is also a full run. Direct-core tests must prove it does not require CLI monkeypatches for stage or evidence output.

## 6. Read structured evidence and CSV without rerunning training

```bash
"$PY" - <<'PY'
import csv, json, os
from pathlib import Path
root = Path(os.environ['DEBUG']) / 'outputs/08_full_pipeline/logs'
logs = list(root.glob('training-*.logs'))
assert len(logs) == 1
with logs[0].open(encoding='utf-8') as handle:
    events = [json.loads(line) for line in handle]
assert events[0]['event'] == 'run_started'
assert events[-1]['event'] == 'run_completed'
manifest_path = logs[0].with_suffix('') / 'manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
assert manifest['audit_run_id'] == events[0]['audit_run_id']
assert manifest['log_complete'] and manifest['coverage_complete']
for entry in manifest['exports']:
    assert entry['status'] == 'complete'
    path = manifest_path.parent / entry['path']
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == entry['columns']
        assert sum(1 for _ in reader) == entry['rows']
print('Run identity and CSV row counts verified')
PY
```

Contract tests additionally check content hashes, field types, identities and invalid file cases; merely printing these counts is not enough. Users may open the referenced CSV in a spreadsheet or use `pandas.read_csv` for larger analysis. There is no planned CSV-replay command.

## 7. Scientific parity and completion evidence

Use baseline, normal and debug directories in automated parity assertions:

- Compare row memberships, raw/encoded feature contracts, selected model/settings and output table schemas.
- Compare scientific numerical columns with `rtol=1e-9, atol=1e-9`; exclude explicit timing columns/IDs/workspace paths. Investigate differences rather than loosening tolerance automatically.
- Compare loaded estimator parameters and existing inference-equivalence evidence, not pickle bytes.
- Unit spies assert identical fit/predict/split calls and inputs with no session, normal session and debug session. Include tuning trials, existing permutation diagnostics and the two reload-check predictions; do not confuse those pre-existing calls with new logging work.
- Require actual terminal events for all coverage stages; verify failure/cancellation leaves no falsely completed child/parent and no final synthetic PASS.
- Verify every displayed metric and count maps to a source record/export at its displayed precision, and each model has an explicit fit-evidence status.
- Record runtime, event count, log/export bytes and the five-minute reviewer exercise. Historic times/results in the prior guide are context only, not verification of this feature.
- Ensure the original project's `outputs/` and `artifacts/` remain unchanged.

Record commands, exit codes, asserted outcomes and limitations in `verification.md`. Run `graphify update .` from the project after verified code changes, or document why it could not run. Planning-only changes do not require a code graph refresh.
