from __future__ import annotations

import math

import pandas as pd
import pytest

from ai_job_market.scenario_policy import (
    ScenarioValidationError,
    build_scenario_policy,
    validate_scenarios,
)


@pytest.fixture
def dev():
    return pd.DataFrame(
        {
            "job_category": ["Research", "Research", "Architecture", "Architecture"],
            "job_title": ["Scientist", "Scientist", "Architect", "Architect"],
            "years_of_experience": [4.0, 9.0, 3.0, 12.0],
        }
    )


@pytest.fixture
def metadata():
    return {
        "category_options": {
            "job_category": ["Architecture", "Research"],
            "job_title": ["Architect", "Scientist"],
        }
    }


def test_policy_uses_observed_pairs_and_preserves_fractional_median(dev, metadata):
    policy = build_scenario_policy(dev, metadata, dataset_id="dev-1")
    assert policy["pair_source"] == "observed_dev"
    assert policy["pairs"] == [
        {"job_category": "Architecture", "job_title": "Architect", "support": 2},
        {"job_category": "Research", "job_title": "Scientist", "support": 2},
    ]
    scientist = policy["titles"]["Scientist"]
    assert scientist == {"min": 4.0, "max": 9.0, "median": 6.5, "support": 2}


def test_policy_rejects_metadata_that_does_not_cover_observed_values(dev, metadata):
    metadata["category_options"]["job_title"] = ["Architect"]
    with pytest.raises(ScenarioValidationError, match="metadata"):
        build_scenario_policy(dev, metadata, dataset_id="dev-1")


@pytest.mark.parametrize("years", [float("nan"), float("inf"), True, "seven"])
def test_manual_scenario_rejects_non_numeric_years(dev, metadata, years):
    policy = build_scenario_policy(dev, metadata, dataset_id="dev-1")
    with pytest.raises(ScenarioValidationError):
        validate_scenarios(
            [
                {
                    "scenario_id": "s1",
                    "job_category": "Research",
                    "job_title": "Scientist",
                    "years_of_experience": years,
                }
            ],
            policy,
        )


def test_manual_scenario_cannot_bypass_bounds(dev, metadata):
    policy = build_scenario_policy(dev, metadata, dataset_id="dev-1")
    row = {
        "scenario_id": "s1",
        "source": "manual",
        "job_category": "Research",
        "job_title": "Scientist",
        "years_of_experience": 12,
    }
    with pytest.raises(ScenarioValidationError, match="outside observed"):
        validate_scenarios([row], policy, allow_exceptions=True)


def test_benchmark_experience_exception_is_explicit_and_bounded(dev, metadata):
    policy = build_scenario_policy(dev, metadata, dataset_id="dev-1")
    row = {
        "scenario_id": "b1",
        "source": "benchmark",
        "record_id": "test:abc:1",
        "job_category": "Research",
        "job_title": "Scientist",
        "years_of_experience": 12,
    }
    with pytest.raises(ScenarioValidationError, match="acknowledgement"):
        validate_scenarios([row], policy, allow_exceptions=True)
    valid = validate_scenarios(
        [row],
        policy,
        allow_exceptions=True,
        acknowledgements={"b1": policy["policy_hash"]},
    )
    assert valid[0]["validation_mode"] == "experience_exception"
    assert valid[0]["exception_reason"] == "experience_outside_observed_dev_range"


def test_unknown_pair_is_never_excepted(dev, metadata):
    policy = build_scenario_policy(dev, metadata, dataset_id="dev-1")
    row = {
        "scenario_id": "b1",
        "source": "benchmark",
        "job_category": "Architecture",
        "job_title": "Scientist",
        "years_of_experience": 7,
    }
    with pytest.raises(ScenarioValidationError, match="observed"):
        validate_scenarios(
            [row], policy, allow_exceptions=True, acknowledgements={"b1": policy["policy_hash"]}
        )


def test_policy_hash_does_not_read_target(dev, metadata):
    with_target = dev.assign(annual_salary_usd=[1, 2, 3, 4])
    changed_target = with_target.assign(annual_salary_usd=[400, 300, 200, 100])
    assert (
        build_scenario_policy(with_target, metadata, dataset_id="same")["policy_hash"]
        == build_scenario_policy(changed_target, metadata, dataset_id="same")["policy_hash"]
    )
