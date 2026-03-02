#!/bin/bash
set -e

echo "Training recommender model..."
uv run python -m src.models.train_model

echo "Starting application..."
uv run uvicorn app.main:app --reload
