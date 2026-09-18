from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "TRAINING_AUDIT.md"
README = ROOT / "README.md"


def test_training_audit_guide_contains_required_sections_and_caveats():
    text = DOC.read_text(encoding="utf-8")

    required = [
        "# Training Audit Guide",
        "## Scope",
        "## Running Normal and Debug Modes",
        "## How to Find Logs",
        "## Log Event Contract",
        "## Data, Feature and Count Evidence",
        "## Fold Evidence",
        "## Model Comparison, Fit Assessment and Tuning",
        "## Current Behavior Findings",
        "## Historical Metrics",
        "## Validation Commands",
        "## Limitations",
    ]
    for heading in required:
        assert heading in text

    assert "R² 0.85 is informational" in text
    assert "historical artifact" in text
    assert "locked-test" in text
    assert "CV" in text
    assert "does not change model selection" in text
    assert "provided_tuning_table_first_row" in text
    assert "--debuglog" in text
    assert "insufficient evidence" in text
    assert "Raw JSON is never printed" in text
    assert "20 data rows" in text


def test_readme_links_training_audit_guide():
    text = README.read_text(encoding="utf-8")
    assert "docs/TRAINING_AUDIT.md" in text
