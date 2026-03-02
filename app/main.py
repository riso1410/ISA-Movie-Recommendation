"""
FastAPI backend for Movie Swipe Recommender.
Tinder-style movie discovery with content-based recommendations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ast import literal_eval
import pickle
import sys

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_features
from src.models.predict_model import ContentBasedRecommender

app = FastAPI(title="Movie Swipe Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
recommender: ContentBasedRecommender | None = None
movies_df: pd.DataFrame | None = None
session: dict = {"liked": [], "disliked": [], "seen": set()}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class SwipeRequest(BaseModel):
    movie_id: int
    direction: str  # "right" (like) or "left" (dislike)


# ---------------------------------------------------------------------------
# Startup – load data & fit recommender
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    global recommender, movies_df

    data_dir = PROJECT_ROOT / "data"

    # Load processed movie data
    processed = pd.read_csv(data_dir / "processed" / "movies_processed.csv")

    # Parse list columns that were saved as strings
    for col in ["genres", "keywords", "cast"]:
        processed[col] = processed[col].apply(
            lambda x: literal_eval(x)
            if isinstance(x, str) and x.startswith("[")
            else []
        )

    # Fill NaN in text columns
    for col in ["decade", "language", "collection", "poster_url"]:
        if col in processed.columns:
            processed[col] = processed[col].fillna("")

    # Load pre-trained model if available, otherwise fit from scratch
    model_path = PROJECT_ROOT / "models" / "recommender.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            recommender = pickle.load(f)
        # Ensure numeric types
        recommender.smd = recommender.smd.reset_index(drop=True)
        for col in ["vote_average", "vote_count", "popularity"]:
            recommender.smd[col] = pd.to_numeric(recommender.smd[col], errors="coerce").fillna(0.0)

        # Merge poster URLs: CSV poster_url first, then fallback to poster_path
        csv_poster = processed.drop_duplicates(subset="id").set_index("id")["poster_url"]
        recommender.smd["poster_url"] = recommender.smd["id"].map(csv_poster).fillna("")

        # Fill remaining blanks from poster_path (raw TMDB metadata)
        if "poster_path" in recommender.smd.columns:
            missing = recommender.smd["poster_url"] == ""
            paths = recommender.smd.loc[missing, "poster_path"].fillna("")
            recommender.smd.loc[missing, "poster_url"] = paths.apply(
                lambda p: f"https://image.tmdb.org/t/p/w500{p}" if p else ""
            )

        movies_df = recommender.smd.copy()
        print(f"Loaded pre-trained model from {model_path}")
    else:
        processed = build_features(processed)
        recommender = ContentBasedRecommender(processed)
        recommender.fit()
        movies_df = processed.reset_index(drop=True)
        print("No pre-trained model found, fitted from scratch")

    print(f"Loaded {len(movies_df)} movies, recommender ready!")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def movie_to_dict(row) -> dict:
    """Convert a DataFrame row to an API-friendly dict."""
    return {
        "id": int(row["id"]),
        "title": str(row["title"]),
        "overview": str(row["overview"])[:500],
        "genres": row["genres"] if isinstance(row["genres"], list) else [],
        "vote_average": round(float(row["vote_average"]), 1)
        if pd.notna(row["vote_average"])
        else 0.0,
        "vote_count": int(row["vote_count"])
        if pd.notna(row["vote_count"])
        else 0,
        "poster_url": str(row.get("poster_url", "")) if pd.notna(row.get("poster_url", "")) else "",
    }


def _get_liked_titles() -> list[str]:
    """Return titles corresponding to the liked movie ids."""
    if not session["liked"]:
        return []
    liked_rows = movies_df[movies_df["id"].isin(session["liked"])]
    return liked_rows["title"].tolist()


def _aggregate_recommendations(liked_titles: list[str], n_per_title: int = 15) -> list[tuple[int, float]]:
    """
    For each liked title, fetch recommendations and aggregate scores.
    Returns list of (movie_id, score) sorted descending.
    """
    seen = session["seen"]
    rec_scores: dict[int, float] = {}

    for title in liked_titles:
        recs = recommender.recommend(title, n=n_per_title)
        for _, row in recs.iterrows():
            rec_title = row["title"]
            mid_rows = movies_df[movies_df["title"] == rec_title]
            if mid_rows.empty:
                continue
            mid = int(mid_rows["id"].values[0])
            if mid in seen:
                continue
            if mid not in rec_scores:
                rec_scores[mid] = 0.0
            rec_scores[mid] += (
                row.get("similarity_score", 0.0)
                + row.get("weighted_rating", 0.0) * 0.1
            )

    return sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/swipe")
def swipe(req: SwipeRequest):
    """Record a user swipe (like / dislike) on a movie."""
    if req.direction == "right":
        session["liked"].append(req.movie_id)
    else:
        session["disliked"].append(req.movie_id)
    session["seen"].add(req.movie_id)
    return {"status": "ok"}


@app.get("/api/movie")
def get_next_movie():
    """Return the next movie card the user should see."""
    seen = session["seen"]
    liked_titles = _get_liked_titles()

    # After 3+ likes, prefer content-based recommendations
    if len(liked_titles) >= 3:
        sorted_recs = _aggregate_recommendations(liked_titles, n_per_title=15)
        for mid, _ in sorted_recs:
            movie_row = movies_df[movies_df["id"] == mid]
            if not movie_row.empty:
                return movie_to_dict(movie_row.iloc[0])

    # Fallback: random unseen movie (weighted by popularity so good movies appear more often)
    unseen = movies_df[~movies_df["id"].isin(seen)]
    if not unseen.empty:
        weights = unseen["popularity"].clip(lower=0.1)
        weights = weights / weights.sum()
        chosen = unseen.sample(n=1, weights=weights)
        return movie_to_dict(chosen.iloc[0])

    return {"error": "No more movies to show!"}


@app.get("/api/recommendations")
def get_recommendations():
    """Return current recommendations based on all liked movies."""
    if not session["liked"]:
        return {"recommendations": [], "liked_count": 0}

    liked_titles = _get_liked_titles()

    # Aggregate recommendations
    seen = session["seen"]
    rec_scores: dict[int, dict] = {}
    for title in liked_titles:
        recs = recommender.recommend(title, n=10)
        for _, row in recs.iterrows():
            rec_title = row["title"]
            mid_rows = movies_df[movies_df["title"] == rec_title]
            if mid_rows.empty:
                continue
            mid = int(mid_rows["id"].values[0])
            if mid in seen:
                continue
            if mid not in rec_scores:
                rec_scores[mid] = {"score": 0.0, "count": 0}
            rec_scores[mid]["score"] += row.get("weighted_rating", 0.0)
            rec_scores[mid]["count"] += 1

    sorted_recs = sorted(
        rec_scores.items(), key=lambda x: x[1]["score"], reverse=True
    )[:20]

    result = []
    for mid, _ in sorted_recs:
        movie_row = movies_df[movies_df["id"] == mid]
        if not movie_row.empty:
            result.append(movie_to_dict(movie_row.iloc[0]))

    return {"recommendations": result, "liked_count": len(session["liked"])}


@app.get("/api/stats")
def get_stats():
    """Return session statistics."""
    # Count actual unique recommendations available
    liked_titles = _get_liked_titles()
    rec_count = 0
    if liked_titles:
        seen = session["seen"]
        rec_ids = set()
        for title in liked_titles:
            recs = recommender.recommend(title, n=10)
            for _, row in recs.iterrows():
                mid_rows = movies_df[movies_df["title"] == row["title"]]
                if not mid_rows.empty:
                    mid = int(mid_rows["id"].values[0])
                    if mid not in seen:
                        rec_ids.add(mid)
        rec_count = len(rec_ids)

    return {
        "liked": len(session["liked"]),
        "disliked": len(session["disliked"]),
        "recommendations_count": rec_count,
    }


@app.get("/api/reset")
def reset_session():
    """Clear all session state."""
    session["liked"] = []
    session["disliked"] = []
    session["seen"] = set()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Static file serving (frontend)
# ---------------------------------------------------------------------------
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_frontend():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Movie Swipe Recommender API. Frontend not yet deployed."}
