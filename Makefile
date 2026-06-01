SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python))
PIP    ?= $(PYTHON) -m pip
PYTEST_ARGS ?=
RUFF_ARGS   ?=
MYPY_ARGS   ?=

ifeq ($(OS),Windows_NT)
MAKE := make
endif

ifneq (,$(findstring xterm,$(TERM)))
    BLUE  := \033[34m
    GREEN := \033[32m
    YELLOW := \033[33m
    RED   := \033[31m
    RESET := \033[0m
else
    BLUE := GREEN := YELLOW := RED := RESET :=
endif

.PHONY: help install clean lint fmt fmt-check typecheck test test-cov check
.PHONY: verify-wheel-data smoke-gate _check-python

help:
	@echo ""
	@echo "$(BLUE)Engrava — Development Commands$(RESET)"
	@echo ""
	@echo "  install             Install package + dev dependencies"
	@echo "  clean               Remove build/cache artifacts"
	@echo ""
	@echo "  lint                Ruff lint check"
	@echo "  fmt                 Ruff format (write changes)"
	@echo "  fmt-check           Ruff format check (read-only)"
	@echo "  typecheck           Mypy strict"
	@echo "  test                pytest (90% cov enforced)"
	@echo "  test-cov            pytest + HTML coverage report"
	@echo "  check               Full quality gate: lint + fmt-check + typecheck + test"
	@echo ""
	@echo "$(GREEN)Pre-release (not part of check — builds wheel + sdist):$(RESET)"
	@echo "  verify-wheel-data   Verify schema_core.sql + synthetic-v1.json bundled in wheel/sdist"
	@echo "  smoke-gate          Pre-publish quality gate (synthetic benchmark binding ACs)"
	@echo ""

install:
	@echo "$(BLUE)>> Installing engrava + dev dependencies$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)>> Done. Run 'make check' to verify quality gate.$(RESET)"

clean:
	rm -rf build/ dist/ src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf .coverage .coverage.* htmlcov coverage.xml
	find . \( -type d -name __pycache__ -o -type f -name '*.pyc' \) -prune -exec rm -rf {} +

lint:
	@echo "$(BLUE)>> Ruff lint$(RESET)"
	$(PYTHON) -m ruff check src/ tests/ $(RUFF_ARGS)

fmt:
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

fmt-check:
	@echo "$(BLUE)>> Ruff format check$(RESET)"
	$(PYTHON) -m ruff format --check src/ tests/
	$(PYTHON) -m ruff check src/ tests/

typecheck:
	@echo "$(BLUE)>> Mypy strict$(RESET)"
	$(PYTHON) -m mypy --strict src/ $(MYPY_ARGS)

test:
	@echo "$(BLUE)>> pytest$(RESET)"
	$(PYTHON) -m pytest --cov=engrava --cov-fail-under=90 $(PYTEST_ARGS)

test-cov:
	$(PYTHON) -m pytest --cov=engrava --cov-report=term-missing --cov-report=html $(PYTEST_ARGS)
	@echo "$(GREEN)>> Coverage HTML: htmlcov/index.html$(RESET)"

# Guard: validate Python interpreter before any quality gate step.
# Catches broken venv (interpreter path exists but binary is missing/corrupt).
_check-python:
	@$(PYTHON) --version >/dev/null 2>&1 || \
		(echo "$(RED)>> Python not found or venv broken. Run 'make install' or 'uv sync'.$(RESET)" && exit 1)

check: _check-python
	@echo "$(BLUE)>> Full quality gate$(RESET)"
	@$(MAKE) lint
	@$(MAKE) fmt-check
	@$(MAKE) typecheck
	@$(MAKE) test
	@echo "$(GREEN)>> Quality gate PASSED$(RESET)"

# Pre-release check — intentionally NOT part of 'check'.
# Runs 'python -m build' (slow) and asserts schema_core.sql + synthetic-v1.json
# are bundled in both the wheel and sdist. Run manually before tagging a release.
verify-wheel-data: _check-python
	@echo "$(BLUE)>> Verifying wheel + sdist package data$(RESET)"
	$(PYTHON) scripts/verify_wheel_data.py
	@echo "$(GREEN)>> verify-wheel-data PASSED$(RESET)"

# Pre-release quality gate — intentionally NOT part of 'check'.
# Drives the bundled synthetic benchmark in binding-AC mode (~5 minutes
# walltime on reference hardware) and checks the committed floors in
# scripts/check_smoke_gate.py. Hard fail blocks the publish workflow.
# Requires the embeddings-local extra (sentence-transformers + torch).
smoke-gate: _check-python
	@echo "$(BLUE)>> Pre-publish smoke gate$(RESET)"
	$(PYTHON) scripts/check_smoke_gate.py
	@echo "$(GREEN)>> smoke-gate PASSED$(RESET)"
