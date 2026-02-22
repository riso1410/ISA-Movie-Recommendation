.PHONY: install serve notebook clean help

## Install dependencies
install:
	uv sync

## Run the MovieMatch web app
serve:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

## Open the Jupyter notebook
notebook:
	uv run jupyter notebook notebooks/01_eda_and_preprocessing.ipynb

## Delete compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Show available commands
help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies (uv sync)"
	@echo "  make serve     - Run the MovieMatch web app on http://localhost:8000"
	@echo "  make notebook  - Open the Jupyter notebook"
	@echo "  make clean     - Delete compiled Python files"

.DEFAULT_GOAL := help
