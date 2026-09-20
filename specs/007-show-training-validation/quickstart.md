# Quickstart: Show Training Validation on Pages 04–05

**Status**: Implemented after explicit task approval; final browser and reader acceptance remain gated.

## 1. Confirm the current offline state

From the repository root:

```bash
.venv/bin/python training_validation.py inspect \
  --workspace . \
  --policy config/training_validation.json
```

The current workspace is expected to return Blocked because no verified unseen reserve exists. This is the expected real UI unavailable state. Do not run training or create a fallback pack.

## 2. Test the read-only consumer

Implementation tests use temporary complete/corrupt fixture packs only:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_training_validation_views.py \
  tests/test_model_ui_pages.py
```

Required assertions include latest-valid selection, invalid-newer fallback to older valid pack, no-pack/corrupt-pack messages, exact fold/model/final values, source independence and zero model/training operations.

## 3. Run the application after implementation

```bash
streamlit run streamlit.py
```

Open:

- `4. Model Comparison`
- `5. Best Model & Importance`

Expected current workspace behavior:

- each page contains collapsed training-validation expander(s);
- expanding them shows `Training validation unavailable` plus the read-only inspection guidance;
- with a valid fixture/future pack, Page 04 shows exact fold row counts, all 25 candidate-fold steps and five candidate conclusions, while Page 05 shows tuning/variant/final steps;
- raw `training.log`/`events.jsonl` are visibly distinct from the detailed evidence-derived transcript and all original logs plus the complete transcript are downloadable;
- existing supplemental page content continues to render;
- the separate Historical Branch B report shows five Page 04 candidate 5W1H records, selected/Full/Top-2 Page 05 records, bounded raw audit events and complete verified downloads from existing evidence;
- historical locked-test exposure is explicit and no historical value fills strict training-validation fields;
- no training, model loading, inference, publication or file write occurs.

If a separately approved future offline pack exists, first validate it outside Streamlit:

```bash
.venv/bin/python training_validation.py check \
  --workspace /absolute/eligible-workspace \
  --run-id tv-0123456789abcdef0123456789abcdef
```

Then launch Streamlit against that established workspace using the existing data-root mechanism. The UI selects the latest valid complete pack; there is no run selector or training button.

## 4. Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
ruff check src/pages/training_validation_presentation.py \
  src/pages/page04_model_comparison.py \
  src/pages/page05_best_model.py \
  tests/test_training_validation_views.py \
  tests/test_model_ui_pages.py
git diff --check
```

Also:

1. compare protected producer/pipeline/Page 06/output/artifact/lock hashes;
2. confirm no root `outputs/training_validation/` pack was generated;
3. inspect both pages through the real entrypoint;
4. perform 1280×800 and 1440×900 browser review if tooling is available, otherwise record pending visual acceptance;
5. run Graphify update after source changes.

## 5. Reader walkthrough

A reader should locate within five minutes:

- exact run ID and generated time, or why unavailable;
- outer fold 3 train/validation months and row counts;
- how history expands and protected populations remain excluded;
- lowest CV MAE versus selected family versus fastest fit;
- all five candidate decisions and their fold-by-fold evidence;
- whether a displayed step is a raw event or an evidence-derived transcript row;
- tuning run/skip status and trial/parent-fold details;
- Full/Top-2 holdout results and non-promotion rule;
- q90 descriptive/not-calibrated caveat;
- inference reserve exclusion;
- safe next action.
