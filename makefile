# Everything runs through uv, so the tools come from the project's venv.
UV ?= uv run

TESTS ?= tests
ARGS ?=

.PHONY: sync test lint fix typecheck check examples

sync:
	uv sync

# Examples:
#   make test
#   make test TESTS=tests/test_conditioning.py ARGS='-k fantasize'
test:
	$(UV) pytest $(TESTS) -q $(ARGS)

lint:
	$(UV) ruff check
	$(UV) ruff format --check

# Fix lint warnings, then format (order matters).
fix:
	-$(UV) ruff check --fix
	$(UV) ruff format

typecheck:
	$(UV) ty check src

check: lint typecheck test

# Runs every example; a few minutes on a CPU.
examples:
	for f in examples/*.py; do echo "== $$f"; $(UV) python $$f || exit 1; done
