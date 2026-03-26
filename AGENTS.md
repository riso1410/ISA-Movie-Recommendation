# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

Content-based movie recommendation system (ISA course Mini-project). Per-field TF-IDF/Count vectorizers + cosine similarity + MMR diversity + IMDB weighted rating. Tinder-style swipe UI (MovieMatch).

## Commands

```bash
uv sync                                        # Install dependencies
uv run uvicorn app.main:app --reload           # Run web app (http://localhost:8000)
uv run python -m src.models.train_model        # Train recommender → models/recommender_<mode>.pkl
uv run python -m src.models.evaluate_model     # Evaluate: Precision@K, NDCG@K, Coverage, Diversity, Novelty
uv run jupyter notebook                        # Run Jupyter notebooks
```

No tests exist. No Docker setup. No linter configured.

## Data

Raw data: manually download from Kaggle ("The Movies Dataset") into `data/raw/`. Required CSVs: `movies_metadata.csv`, `credits.csv`, `keywords.csv`, `links_small.csv`. Processed output: `data/processed/movies_processed.csv`. The `data/` directory is gitignored.

## Architecture

### Pipeline: `src/`

```
src/data/make_dataset.py       → Load 5 CSVs, parse JSON columns, clean IDs, merge on movie ID
src/features/build_features.py → Per-field text cleaning: overview_clean, genres_str, keywords_str, cast_str, director_str, decade, language, collection
src/models/predict_model.py    → ContentBasedRecommender: per-field vectorizers → weighted sparse concat → cosine sim → MMR reranking
src/models/train_model.py      → Orchestrates: make_dataset → build_features → fit recommender → pickle
src/models/evaluate_model.py   → Precision@5, NDCG@5, MAP@5, Per-Genre Precision, grid search over weights
```

**Multi-vectorizer approach** (not single "soup"): Each field gets its own TF-IDF or Count vectorizer, weighted separately, then horizontally stacked into one sparse feature matrix. Field weights defined in `DEFAULT_WEIGHTS` dict in `predict_model.py`.

**Name cleaning**: Spaces removed from names ("Johnny Depp" → "johnnydepp") so TF-IDF treats them as single tokens.

**Weighted rating** (IMDB formula): `WR = (v/(v+m))*R + (m/(v+m))*C` where m=60th percentile vote_count, C=mean vote_average.

**Scoring**: `alpha * cosine_similarity + (1-alpha) * normalized_weighted_rating`, then MMR re-ranking for diversity.

### Web App: `app/`

- `app/main.py` — FastAPI backend. Loads prebuilt pickled recommenders at startup and can switch between loaded models. Single-process in-memory session state (liked/disliked/seen) — no multi-user support.
- `app/static/index.html` — React 18 via CDN (no build step). Single-file SPA with Babel JSX transform.

**Movie selection**: Random (popularity-weighted) until 3+ likes, then content-based recs aggregated from all liked movies.

**Endpoints**: `POST /api/swipe`, `GET /api/movie`, `GET /api/recommendations`, `GET /api/stats`, `GET /api/reset`

### Notebook

`notebooks/experiments.ipynb` — Full EDA, preprocessing, 3-iteration modeling, evaluation with Precision@K, and `poster_url` export.

### Design Docs

`docs/evaluation_report.md` — Evaluation results and analysis.
`learn/` — Educational markdown docs (00-06) explaining each system component.
`reports/figures/` — Generated evaluation/EDA visualizations (PNGs).
