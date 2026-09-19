# Quickstart: Validate Training Logs After Approval

These are planned verification commands, not evidence of completed execution. Do not run training until implementation and isolated validation are approved. No command below aims to reach R² 0.85.

## 1. Inspect scope and environment

From the repository root:

```bash
git status --short
.venv/bin/python --version
```

Preserve the outstanding user edit to `requirements.txt`. No installation or upgrade is part of this feature. Record baseline code/config/data hashes and installed package versions before instrumentation; historical output files alone are not a verified baseline.

## 2. Run bounded tests (after the planned tests exist)

```bash
.venv/bin/python -m pytest tests/test_training_behavior.py -q
.venv/bin/python -m pytest tests/test_training_audit.py tests/test_training_fold_audit.py -q
.venv/bin/python -m pytest tests/test_training_audit_integration.py tests/test_training_audit_docs.py -q
.venv/bin/python -m pytest -q
```

Expected evidence: initial characterization passes on original logic; new logging assertions fail before instrumentation and pass afterward. Integration tests must exercise real entrypoints with temporary workspaces/copied roots, never the baseline output directory. Bounded fake-estimator tests do not replace the approved real offline run.

## 3. Approved full core run in an isolated workspace

This invokes the existing full workflow, including segmentation and tuning, and can be expensive. It does not replace root release artifacts.

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from tempfile import mkdtemp
from ai_job_market.core import run_pipeline
root = Path.cwd().resolve()
workspace = Path(mkdtemp(prefix="training-audit-core-"))
print(f"Validation workspace: {workspace}")
result = run_pipeline(
    raw_path=root / "data/raw/ai_jobs_market_2025_2026.csv",
    root=root,
    workspace_root=workspace,
)
print(f"Audit logs: {workspace / 'outputs/08_full_pipeline/logs'}")
print(f"Run returned: {type(result).__name__}")
PY
```

Before implementation, use the same isolation approach to capture an approved baseline if fixture characterization is insufficient. Baseline and instrumented results must use the same source, configuration, seed, and installed environment. Do not run repeatedly to improve scores. Existing locked-test exposure must be acknowledged.

## 4. Actual terminal CLI without overwriting release outputs

The CLI currently has no workspace argument. Copy the minimum project inputs to a temporary project and execute its actual entrypoint using the existing environment:

```bash
.venv/bin/python - <<'PY'
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
root = Path.cwd().resolve()
copy = Path(tempfile.mkdtemp(prefix="training-audit-cli-"))
shutil.copy2(root / "pipeline.py", copy / "pipeline.py")
for name in ("src", "config", "data"):
    shutil.copytree(root / name, copy / name, ignore=shutil.ignore_patterns("__pycache__"))
print(f"CLI validation copy: {copy}", flush=True)
with (copy / "terminal.txt").open("w", encoding="utf-8") as capture:
    result = subprocess.run(
        [sys.executable, str(copy / "pipeline.py"), "--no-heartbeat"],
        cwd=copy, stdout=capture, stderr=subprocess.STDOUT, check=False,
    )
print(f"Exit status: {result.returncode}; terminal capture: {copy / 'terminal.txt'}")
raise SystemExit(result.returncode)
PY
```

A copied root has no Git metadata: provenance must report that explicitly, not fabricate a revision. Record the source revision/hash in the validation report. Unit/integration fixtures separately exercise CLI preflight (2), error (1), and cancellation (130) without repeating full model searches.

## 5. Reconcile evidence

Use test assertions and document their outputs, not console-only observations:

- Parse every `.logs` line as strict JSON; compare audit records with matching terminal records by `audit_run_id` and `sequence`.
- Reconstruct fold position lists; assert actual counts, bounds, digest identity, and consistency with consumed training arrays.
- Verify trial parameters, ranking direction, fixed depth 20 and initial-grid final selection match original behavior; no “fixes” are allowed.
- Compare scientific before/after values: exact memberships/parameters/call counts/schemas; predictions and metrics initially `rtol=1e-9, atol=1e-9`. Exclude durations and generated IDs. Investigate failures before changing tolerances.
- Confirm preprocessing stays fitted on the same training rows and logging adds no held-out prediction/fit calls.
- Confirm original root artifacts are untouched, logging failure is visible, no handlers leak on repeat invocation, and every executed operation has a terminal status unless forcibly killed.
- Record log size and measured overhead with method/limitations; no unmeasured speed claim.
- Update `docs/TRAINING_AUDIT.md` with commands, workspaces, provenance, test outcomes, metric sources and limits. Mark unrun checks explicitly.

## 6. Final review

```bash
git diff --check
git diff --stat
graphify update .
```

Inspect the actual diff for modeling changes and English-only additions. If graph refresh cannot run, record the reason and retain source-verified query evidence. Stop for review on any scope expansion, extra test use, dependency change, or scientific behavior difference.
