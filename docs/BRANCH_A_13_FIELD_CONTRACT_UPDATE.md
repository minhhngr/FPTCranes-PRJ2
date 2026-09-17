# Branch A — Stage-3 13-Field Canonical Input Contract

## Canonical Branch-A input set
Branch A starts only from the Stage-3 **Primary keep (13)** contract:

1. `job_title`
2. `job_category`
3. `years_of_experience`
4. `education_required`
5. `city`
6. `country`
7. `remote_work`
8. `company_size`
9. `industry`
10. `demand_score`
11. `benefits_score_10`
12. `required_skills`
13. `skill_count` — engineered from `required_skills`

This is **12 retained source/business features + 1 engineered feature = 13 Branch-A input fields**.

## Explicitly blocked from Branch A
`experience_level`, `ai_salary_premium_pct`, `demand_growth_yoy_pct`, salary fields, temporal fields, identifiers, flags and all other Stage-3 blocked fields must not enter any R0-R4 representation.

## R0-R4 contract
- **R0** Full Global PCA: 13 fields
- **R1** Family-wise PCA: 13 fields
- **R2** No `job_category`: 12 fields
- **R3** No `years_of_experience`: 12 fields
- **R4** No `job_category` + no `years_of_experience`: 11 fields

Every ablation is therefore a subset of the same Stage-3 source-of-truth contract.

## Files in this patch
- REPLACE `src/ai_job_market/core.py`
- REPLACE `src/pages/page03_segmentation.py`
- REPLACE/ADD `tests/test_segmentation_representation_robustness.py`
- ADD `docs/BRANCH_A_13_FIELD_CONTRACT_UPDATE.md`

`src/ai_job_market/segmentation_robustness.py` does **not** require a logic change for this contract because it consumes `all_raw_features` supplied by `core.py`. `config/project.yaml` also does not require a change.
