# Quickstart validation

1. Run focused RED/GREEN tests for partitions and search:
   ```bash
   .venv/bin/pytest tests/test_training_partitions_v2.py tests/test_training_search_v2.py -q
   ```
2. Run producer contract tests with the synthetic eligible fixture:
   ```bash
   .venv/bin/pytest tests/test_training_validation_cli.py tests/test_training_validation_views.py -q
   ```
3. Confirm the current real workspace is blocked rather than retrained:
   ```bash
   .venv/bin/python training_validation.py inspect --workspace .
   ```
   Expected: `RESERVE_KNOWN_EXPOSED` (or another explicit safe blocker), zero fits/predictions.
4. Run Page 04–06 regression/AppTests:
   ```bash
   .venv/bin/pytest tests/test_model_ui_pages.py tests/test_salary_inference.py -q
   ```
5. Before any real training, obtain a fresh eligible source, non-exposure proof, and exact approval. Then run the offline CLI only; never from Streamlit.
6. Complete quality checks:
   ```bash
   .venv/bin/pytest -q
   .venv/bin/ruff check src tests
   git diff --check
   graphify . --update --no-viz
   ```
