ISA Movie Recommendation
==============================

Content-based movie recommender system for ISA course (Mini-project 1).

Uses TF-IDF vectorization on movie metadata (genres, keywords, cast, director, plot overview) with cosine similarity and IMDB weighted rating filter to recommend similar movies.

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Download dataset

Download [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle and place these files in `data/raw/`:

- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`
- `links_small.csv`
- `ratings_small.csv`

### 3. Run the Jupyter notebook

```bash
uv run jupyter notebook notebooks/01_eda_and_preprocessing.ipynb
```

This notebook contains the full pipeline: EDA, preprocessing, iterative modeling (3 iterations), and evaluation.

### 4. Run the MovieMatch web app

Start the FastAPI backend (which also serves the React frontend):

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

The app loads the recommender model at startup (~15 seconds), then you can:
- Swipe right (like) or left (dislike) on movie cards
- Use arrow keys or the buttons to swipe
- Drag cards with your mouse
- After 3+ likes, personalized recommendations appear in the right panel

## Project Organization

    ├── app
    │   ├── main.py            <- FastAPI backend (API + static file serving)
    │   └── static
    │       └── index.html     <- React frontend (Tinder-style swipe UI)
    │
    ├── data
    │   ├── processed          <- Cleaned and merged movie data
    │   └── raw                <- Original CSVs from Kaggle
    │
    ├── docs
    │   └── plans              <- Design and implementation documents
    │
    ├── models                 <- Trained models (pickle files)
    │
    ├── notebooks
    │   └── 01_eda_and_preprocessing.ipynb  <- Full EDA + modeling + evaluation
    │
    ├── reports
    │   └── figures            <- Generated visualizations (8 PNG files)
    │
    ├── src                    <- Reusable Python pipeline modules
    │   ├── data
    │   │   └── make_dataset.py       <- Data loading, cleaning, merging
    │   ├── features
    │   │   └── build_features.py     <- Text cleaning, metadata soup
    │   └── models
    │       ├── predict_model.py      <- ContentBasedRecommender class
    │       └── train_model.py        <- Training script
    │
    ├── pyproject.toml
    └── uv.lock

## Tech Stack

- **uv** for dependency management
- **Python 3.12**, pandas, numpy, scikit-learn
- **FastAPI** + uvicorn (backend)
- **React 18** via CDN (frontend, single HTML file)
- **matplotlib**, seaborn, wordcloud (visualizations)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
