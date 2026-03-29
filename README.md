# ISA Movie Recommendation

Content-based movie recommendation system for ISA course (Mini-project 1).

Multi-vectorizer TF-IDF/Count approach with per-field weighting, cosine similarity, IMDB weighted rating blending, and MMR diversity re-ranking. Includes a Tinder-style swipe UI (MovieMatch) with real-time user profile building.

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

### 2. Download dataset

Download [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle and place these files in `data/raw/`:

- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`
- `links_small.csv`
- `ratings_small.csv` (required for ratings-based evaluation)

### 3. Train recommender models

```bash
uv run python -m src.models.train_model
```

Trains 3 model variants and saves pickle files to `models/`:
- `recommender_overview.pkl` — TF-IDF on plot text only (baseline)
- `recommender_soup.pkl` — TF-IDF on concatenated metadata soup
- `recommender_multi.pkl` — Per-field vectorizers with separate weights (default)

You can also run the `experiments.ipynb` to train models

### 4. Run the MovieMatch web app

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**. Features:
- Swipe right/left on movie cards (mouse drag, buttons, or arrow keys)
- After 3+ likes, personalized recommendations appear in the right panel
- Switch between model variants from the UI
- Analytics tab shows offline evaluation metrics and live session metrics

### 5. Run offline evaluation

```bash
uv run python -m src.models.evaluate_model
```

Runs two evaluation methods. Reports:
- **Ratings-based (leave-N-out):** Precision@K, Recall@K, NDCG@K, MAP@K, MRR, HR@K at K=5 and K=10, plus Coverage, ILD, Novelty, Serendipity
- **Calibration (Steck 2018):** KL-divergence, Jensen-Shannon divergence, Calibration Score — measures whether recommendation genre distribution matches user preferences

Optional grid search over field weights:

```bash
uv run python -m src.models.evaluate_model --grid
```

### 6. Run the Jupyter notebook

```bash
uv run jupyter notebook notebooks/experiments.ipynb
```

Full pipeline: EDA, preprocessing, 3-iteration modeling, method comparison, and quantitative evaluation with all metrics.

## All Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies |
| `uv run python -m src.models.train_model` | Train all 3 recommender models |
| `uv run python -m src.models.evaluate_model` | Run offline evaluation (Precision, NDCG, MAP, MRR, HR, Coverage, ILD, Novelty, Serendipity) |
| `uv run python -m src.models.evaluate_model --grid` | Grid search over field weights |
| `uv run uvicorn app.main:app --host 0.0.0.0 --port 8000` | Start the web app |
| `uv run uvicorn app.main:app --reload` | Start web app with auto-reload (development) |
| `uv run jupyter notebook` | Launch Jupyter notebooks |

## Project Structure

```
├── app/
│   ├── main.py                  # FastAPI backend (API + static file serving)
│   └── static/
│       └── index.html           # React 18 frontend (single-file SPA, no build step)
│
├── data/
│   ├── raw/                     # Original CSVs from Kaggle (gitignored)
│   └── processed/               # Cleaned and merged movie data (gitignored)
│
├── models/                      # Prebuilt recommender pickle files (gitignored)
│
├── notebooks/
│   └── experiments.ipynb        # Full EDA + modeling + evaluation notebook
│
├── reports/
│   └── figures/                 # Generated evaluation and EDA visualizations
│
├── src/
│   ├── data/
│   │   └── make_dataset.py      # Data loading, cleaning, merging
│   ├── features/
│   │   └── build_features.py    # Per-field text cleaning and feature engineering
│   └── models/
│       ├── predict_model.py     # ContentBasedRecommender class
│       ├── train_model.py       # Training script (all 3 modes)
│       └── evaluate_model.py    # Evaluation: ratings-based + genre-based + grid search
│
├── EXPLAINED.md                 # Detailed explanation of how the system works
├── CLAUDE.md                    # AI assistant instructions
├── pyproject.toml               # Project metadata and dependencies
└── uv.lock                      # Locked dependency versions
```

## Tech Stack

- **Python 3.12**
- **uv** 
- **FastAPI**
- **React 18**

## Evaluation Methodology

**Primary (ratings-based):** Leave-N-out evaluation with real MovieLens user ratings. For each of 200 users, 70% of liked movies (rating >= 4.0) form the profile, 30% are held out as ground truth. Metrics: Precision@K, Recall@K, NDCG@K, MAP@K, MRR, HR@K, Coverage, ILD, Novelty, Serendipity.

**Calibration (Steck 2018):** Measures whether the genre distribution of recommendations matches the user's preference distribution. Uses Jensen-Shannon divergence (bounded [0,1]) and KL-divergence. Catches failure modes like a drama/comedy user getting 100% drama recs despite high precision.

See [EXPLAINED.md](EXPLAINED.md) for a detailed walkthrough of the entire pipeline.
