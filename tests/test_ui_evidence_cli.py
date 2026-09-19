from __future__ import annotations

from pathlib import Path

from ai_job_market.ui_evidence import check_workspace


def test_check_workspace_reports_missing_without_writing(tmp_path):
    result = check_workspace(tmp_path)
    assert result["valid"] is False
    assert "missing" in result["message"].lower()
    assert not (tmp_path / "outputs").exists()
