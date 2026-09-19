from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ai_job_market.ui_evidence_io import load_current_evidence


def test_real_supplemental_pack_is_readable_and_top2_reloads():
    raw = os.getenv("UI_EVIDENCE_WORKSPACE")
    if not raw:
        pytest.skip("set UI_EVIDENCE_WORKSPACE to run the saved-pack contract check")
    root = Path(raw).resolve()
    manifest = load_current_evidence(root)
    run = root / "outputs/ui_evidence" / manifest["evidence_id"]
    variants = pd.read_csv(run / "variant_test_predictions.csv")
    full = variants[variants.model_id == "full:selected"].record_id.tolist()
    top2 = variants[variants.model_id == "top2:fixed"].record_id.tolist()
    assert full == top2
    benchmarks = pd.read_csv(run / "benchmark_examples.csv")
    assert set(benchmarks.record_id) <= set(full)
    assert benchmarks.historically_exposed.all()
    assert not benchmarks.pristine.any()
    meta_path = root / "artifacts/ui_evidence" / manifest["evidence_id"] / "top2_metadata.json"
    model_path = meta_path.with_name("top2_model.joblib")
    metadata = json.loads(meta_path.read_text())
    model = joblib.load(model_path)
    test = pd.read_csv(root / "outputs/02_data_ready_for_ml/locked_test_raw.csv")
    prediction = np.asarray(model.predict(test[metadata["feature_order"]].head(3)), dtype=float)
    saved = (
        variants[variants.model_id == "top2:fixed"]
        .sort_values("record_offset")
        .predicted_salary_usd.head(3)
        .to_numpy()
    )
    np.testing.assert_allclose(prediction, saved, rtol=1e-12, atol=1e-12)
