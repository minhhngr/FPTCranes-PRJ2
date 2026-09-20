from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from pages.page05_best_model import _branch_b_fold_table

ROOT = Path(__file__).resolve().parents[1]


def test_branch_b_fold_table_keeps_one_evidence_backed_row_per_fold() -> None:
    folds = pd.DataFrame(
        {
            "model": ["A", "B", "A", "B"],
            "fold_id": ["fold-1", "fold-1", "fold-2", "fold-2"],
            "train_period": ["2025-01..2025-04", "2025-01..2025-04", "2025-05..2025-08", "2025-05..2025-08"],
            "validation_period": ["2025-05..2025-08", "2025-05..2025-08", "2025-09..2025-12", "2025-09..2025-12"],
            "train_rows": [200, 200, 200, 200],
            "validation_rows": [200, 200, 201, 201],
        }
    )

    result = _branch_b_fold_table(folds)

    assert list(result.columns) == [
        "Step",
        "Fold",
        "Flow",
        "Train period",
        "Validation period",
        "Train rows",
        "Validation rows",
    ]
    assert result.to_dict("records") == [
        {
            "Step": 1,
            "Fold": "fold-1",
            "Flow": "Train → Validation",
            "Train period": "2025-01..2025-04",
            "Validation period": "2025-05..2025-08",
            "Train rows": 200,
            "Validation rows": 200,
        },
        {
            "Step": 2,
            "Fold": "fold-2",
            "Flow": "Train → Validation",
            "Train period": "2025-05..2025-08",
            "Validation period": "2025-09..2025-12",
            "Train rows": 200,
            "Validation rows": 201,
        },
    ]


def test_page05_shows_branch_b_fold_method_before_tuning_detail() -> None:
    app = AppTest.from_file(str(ROOT / "streamlit.py"), default_timeout=30)
    app.session_state["auth_role"] = "admin"
    app.run()
    assert not app.exception
    app.radio[0].set_value("5. Best Model & Importance").run()
    assert not app.exception

    labels = [item.label for item in app.expander]
    assert "Branch B · How the temporal DEV folds are split" in labels
    assert "Priority guide: How hyperparameter tuning works" in labels
    assert labels.index("Branch B · How the temporal DEV folds are split") < labels.index(
        "Priority guide: How hyperparameter tuning works"
    )

    frames = [item.value for item in app.dataframe]
    fold_tables = [
        frame
        for frame in frames
        if {
            "Step",
            "Fold",
            "Flow",
            "Train period",
            "Validation period",
            "Train rows",
            "Validation rows",
        }
        <= set(frame.columns)
    ]
    assert len(fold_tables) == 1
    assert len(fold_tables[0]) == 5

    markdown = [item.value for item in app.markdown]
    assert "**How the fold split works**" in markdown
    assert "**Step-by-step fold execution**" in markdown
    assert "**Observed Branch B training-log events**" in markdown
    assert "**Limits**" in markdown
    assert any("Step 1" in value and "rows" in value for value in markdown)
    assert any("not an expanding-window" in value for value in markdown)
