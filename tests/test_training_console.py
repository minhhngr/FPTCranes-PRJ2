from io import StringIO

from ai_job_market.training_console import ConsoleRenderer, write_block


def _event(event: str, detail_level: int, **extra):
    return {
        "event": event,
        "detail_level": detail_level,
        "level": "INFO",
        "status": "completed",
        "message": "Readable message.",
        "step_id": "unit.step",
        "operation": "unit",
        **extra,
    }


def test_normal_and_debug_depth_use_two_space_hierarchy():
    normal = ConsoleRenderer(debuglog=False)
    debug = ConsoleRenderer(debuglog=True)

    assert normal.render(_event("stage_completed", 1, stage_id="B4")).startswith("[COMPLETED]")
    assert normal.render(_event("data_summary", 2)).startswith("  ")
    assert normal.render(_event("candidate_fold_scored", 3, fold_id=1)) is None
    assert debug.render(_event("candidate_fold_scored", 3, fold_id=1)).startswith("    ")
    assert debug.render(_event("split_defined", 4, split_id="abc")).startswith("      ")


def test_warning_is_visible_regardless_of_depth_and_never_dumps_json():
    renderer = ConsoleRenderer(debuglog=False)
    text = renderer.render(
        _event(
            "logging_degraded",
            4,
            level="WARNING",
            message="Evidence file could not be written.",
            secret="must-not-render",
        )
    )

    assert text is not None
    assert "WARNING" in text
    assert "Evidence file could not be written" in text
    assert "secret" not in text
    assert not text.lstrip().startswith("{")


def test_metric_summary_has_units_and_insufficient_evidence_reason():
    renderer = ConsoleRenderer(debuglog=False)
    text = renderer.render(
        _event(
            "evaluation_completed",
            2,
            model_name="Random Forest",
            MAE_mean=1234.5,
            RMSE_mean=2000.0,
            R2_mean=0.75,
            MedAE_mean=900.0,
            fit_assessment="insufficient_evidence",
            fit_reason="training_scores_not_computed; diagnostic_rule_not_defined",
        )
    )

    assert text is not None
    assert "Random Forest" in text
    assert "MAE=1,234.500 USD" in text
    assert "R²=0.750 (unitless)" in text
    assert "insufficient evidence" in text
    assert "training scores" in text.lower()


def test_write_block_keeps_multiline_output_together():
    stream = StringIO()
    write_block(stream, "first\nsecond")
    assert stream.getvalue() == "first\nsecond\n"
