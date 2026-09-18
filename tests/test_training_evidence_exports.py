import csv
import json
from io import StringIO

from ai_job_market.training_audit import start_training_audit


def test_export_table_keeps_all_rows_and_manifest_identity(tmp_path):
    stream = StringIO()
    rows = [{"ordinal": i, "score": i / 7} for i in range(21)]

    with start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=42,
        target="annual_salary_usd",
        features=["years_of_experience"],
        stream=stream,
    ) as audit:
        export = audit.export_table("trial results", rows, kind="tuning_trials", detail_level=4)
        run_id = audit.audit_run_id
        manifest_path = audit.evidence_dir / "manifest.json"

    assert export["rows"] == 21
    assert export["shown_rows"] == 20
    assert export["omitted_rows"] == 1
    assert export["path"] == "trial-results.csv"
    with (manifest_path.parent / export["path"]).open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 21
    assert float(saved[-1]["score"]) == rows[-1]["score"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["audit_run_id"] == run_id
    assert manifest["training_status"] == "completed"
    assert manifest["exports"][0]["sha256"]
    assert manifest["exports"][0]["columns"] == ["ordinal", "score"]


def test_export_zero_rows_writes_a_header_when_columns_are_declared(tmp_path):
    with start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=1,
        target="target",
        features=[],
        stream=StringIO(),
    ) as audit:
        export = audit.export_table(
            "empty",
            [],
            kind="metric_details",
            columns=["metric", "value"],
        )
        path = audit.evidence_dir / export["path"]

    assert path.read_text(encoding="utf-8").strip() == "metric,value"


def test_export_names_are_generated_not_path_fragments(tmp_path):
    with start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=1,
        target="target",
        features=[],
        stream=StringIO(),
    ) as audit:
        export = audit.export_table("../../unsafe name", [{"x": 1}], kind="metric_details")
        assert export["path"] == "unsafe-name.csv"
        assert (audit.evidence_dir / export["path"]).is_file()
