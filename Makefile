VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv

.PHONY: venv lint format test

venv:
	python3.13 -m venv $(VENV)
	$(UV) pip install --python $(PYTHON) ruff pytest

lint:
	$(VENV)/bin/ruff check src/

format:
	$(VENV)/bin/ruff check --fix src/
	$(VENV)/bin/ruff format src/

test:
	$(VENV)/bin/pytest
