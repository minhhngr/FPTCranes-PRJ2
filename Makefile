.PHONY: app install ruff_check clean clean_all test training ui_evidence ui_evidence_check

# ── Configurable paths (override on command line: make training DATA=other.csv)
DATA       ?= data/raw/ai_jobs_market_2025_2026.csv
WORKSPACE  ?= .

# ── Application ───────────────────────────────────────────────────────────────
app:
	uv run streamlit run streamlit.py

ruff_check:
	uv run ruff check .
	uv run ruff format .

test:
	uv run pytest -q

# ── Offline pipeline (training + artifacts) ───────────────────────────────────
# Cross-platform script: also runnable as  python scripts/run_training.py
training:
	uv run python scripts/run_training.py --data "$(DATA)" $(if $(filter-out .,$(WORKSPACE)),--workspace "$(WORKSPACE)")

# ── Supplemental diagnostic evidence for pages 04/05 ─────────────────────────
# Cross-platform script: also runnable as  python scripts/run_ui_evidence.py
ui_evidence:
	uv run python scripts/run_ui_evidence.py --workspace "$(abspath $(WORKSPACE))"

ui_evidence_check:
	uv run python scripts/run_ui_evidence.py --check --workspace "$(abspath $(WORKSPACE))"

clean:
	rm -rf .pytest_cache .logs/* .ruff_cache .mypy_cache .coverage* allure-* allure-results allure-report htmlcov dist build *.egg-info *:Zone.Identifier
	find . -type d -name ".venv" -prune -o -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.Identifier" -o -name "*:Zone.Identifier" \) -delete

clean_all:
	rm -rf .pytest_cache .venv .internallogs .logs attachments/* .ruff_cache .mypy_cache .coverage* allure-* allure-results allure-report htmlcov dist build *.egg-info migration_*.db
	find . -type d -name ".venv" -prune -o -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.Identifier" -o -name "*:Zone.Identifier" \) -delete
