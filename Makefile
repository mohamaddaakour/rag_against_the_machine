UV := uv
ARGS ?=
MYPY_FLAGS := --warn-return-any --warn-unused-ignores \
              --ignore-missing-imports --disallow-untyped-defs \
              --check-untyped-defs

.DEFAULT_GOAL := help
.PHONY: install run debug clean lint lint-strict test index search serve help

install:
	$(UV) sync

run:
	$(UV) run python -m src $(ARGS)

debug:
	$(UV) run python -m pdb -m src $(ARGS)

lint:
	$(UV) run flake8 .
	$(UV) run mypy . $(MYPY_FLAGS)

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

test:
	$(UV) run pytest -q

index:
	$(UV) run python -m src index --max_chunk_size $(or $(SIZE),2000)

search:
	$(UV) run python -m src search "$(Q)" --k $(or $(K),5)

serve:
	$(UV) run python -m src serve --port $(or $(PORT),8000)

clean:
	$(UV) run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts and 'data' not in p.parts]"
	$(UV) run python -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ('.mypy_cache', '.pytest_cache', '.ruff_cache')]"

help:
	@echo "install run debug lint lint-strict test index search serve clean"
