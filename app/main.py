"""
FastAPI backend for Movie Swipe Recommender.
Tinder-style movie discovery with content-based recommendations.
"""

import os
import pickle
import random
import sys
import time
from pathlib import Path
from ast import literal_eval

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict_model import ContentBasedRecommender, MODES, MODE_LABELS

app = FastAPI(title="Movie Swipe Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender: ContentBasedRecommender | None = None
models: dict[str, ContentBasedRecommender] = {}
active_mode: str = "multi"
movies_df: pd.DataFrame | None = None
session: dict = {"liked": [], "disliked": [], "seen": set(), "swipe_log": []}
pending_movie_meta: dict = {
    "source": "random",
    "similarity_score": None,
    "combined_score": None,
}
offline_cache: dict | None = None
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_ENV = "MOVIEMATCH_DEFAULT_MODEL"
MODEL_PREFIX = "recommender_"

FEEDBACK_WEIGHTS: dict[tuple[str, str], float] = {
    ("right", "random"): 1.0,
    ("left", "random"): -0.3,
    ("right", "model"): 2.5,
    ("left", "model"): -1.2,
    ("right", "rec_list"): 3.0,
    ("left", "rec_list"): -1.5,
}
EXPLORATION_RATE = 0.15


class SwipeRequest(BaseModel):
    movie_id: int
    direction: str


class RecFeedbackRequest(BaseModel):
    movie_id: int
    direction: str


def _prepare_model(rec: ContentBasedRecommender, processed: pd.DataFrame) -> None:
    """Attach poster URLs and ensure numeric types on a loaded model."""
    rec.smd = rec.smd.reset_index(drop=True)
    for col in ["vote_average", "vote_count", "popularity"]:
        rec.smd[col] = pd.to_numeric(rec.smd[col], errors="coerce").fillna(0.0)

    csv_poster = processed.drop_duplicates(subset="id").set_index("id")["poster_url"]
    rec.smd["poster_url"] = rec.smd["id"].map(csv_poster).fillna("")


def _load_processed_movies() -> pd.DataFrame:
    processed = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "movies_processed.csv"
    )

    if "poster_url" not in processed.columns:
        processed["poster_url"] = ""

    for col in ["genres", "keywords", "cast"]:
        processed[col] = processed[col].apply(
            lambda x: literal_eval(x)
            if isinstance(x, str) and x.startswith("[")
            else []
        )

    for col in ["decade", "language", "collection", "poster_url"]:
        if col in processed.columns:
            processed[col] = processed[col].fillna("")

    return processed


def _load_models(processed: pd.DataFrame) -> dict[str, ContentBasedRecommender]:
    loaded_models: dict[str, ContentBasedRecommender] = {}

    for path in sorted(MODEL_DIR.glob(f"{MODEL_PREFIX}*.pkl")):
        mode = path.stem.removeprefix(MODEL_PREFIX)
        if not mode:
            continue

        with open(path, "rb") as f:
            rec = pickle.load(f)
        _prepare_model(rec, processed)
        rec.mode = mode
        loaded_models[mode] = rec
        print(f"Loaded {mode} model from {path}")

    return loaded_models


def _pick_active_mode(available_models: dict[str, ContentBasedRecommender]) -> str:
    requested_mode = os.getenv(DEFAULT_MODEL_ENV)
    if requested_mode:
        if requested_mode not in available_models:
            available = ", ".join(sorted(available_models))
            raise RuntimeError(
                f"{DEFAULT_MODEL_ENV}={requested_mode!r} is unavailable. "
                f"Available models: {available}"
            )
        return requested_mode

    if "multi" in available_models:
        return "multi"
    return sorted(available_models)[0]


def _reset_session_state() -> None:
    global pending_movie_meta, offline_cache
    session["liked"] = []
    session["disliked"] = []
    session["seen"] = set()
    session["swipe_log"] = []
    pending_movie_meta = {
        "source": "random",
        "similarity_score": None,
        "combined_score": None,
    }
    offline_cache = None


def _get_recommender() -> ContentBasedRecommender:
    if recommender is None:
        raise RuntimeError("Recommender is not loaded")
    return recommender


def _get_movies_df() -> pd.DataFrame:
    if movies_df is None:
        raise RuntimeError("Movies dataframe is not loaded")
    return movies_df


@app.on_event("startup")
def startup():
    global recommender, models, active_mode, movies_df

    processed = _load_processed_movies()
    models = _load_models(processed)
    if not models:
        raise RuntimeError(
            "No prebuilt recommender models found in models/. "
            "Generate one or more recommender_<mode>.pkl files before starting the app."
        )

    active_mode = _pick_active_mode(models)
    recommender = models[active_mode]
    movies_df = recommender.smd.copy()
    loaded_movie_count = len(recommender.smd)

    print(f"Loaded {len(models)} model(s): {list(models.keys())}")
    print(f"Active model: {active_mode} ({loaded_movie_count} movies)")


def movie_to_dict(row, source: str | None = None) -> dict:
    """Convert a DataFrame row to an API-friendly dict."""
    d = {
        "id": int(row["id"]),
        "title": str(row["title"]),
        "overview": str(row["overview"])[:500],
        "genres": row["genres"] if isinstance(row["genres"], list) else [],
        "vote_average": round(float(row["vote_average"]), 1)
        if pd.notna(row["vote_average"])
        else 0.0,
        "vote_count": int(row["vote_count"]) if pd.notna(row["vote_count"]) else 0,
        "poster_url": str(row.get("poster_url", ""))
        if pd.notna(row.get("poster_url", ""))
        else "",
    }
    if source is not None:
        d["source"] = source
    return d


def _movie_id_to_idx(movie_id: int) -> int | None:
    """Map movie ID to recommender.smd row index."""
    current_recommender = _get_recommender()
    matches = current_recommender.smd.index[current_recommender.smd["id"] == movie_id]
    return int(matches[0]) if len(matches) > 0 else None


def _profile_recommendations(
    n: int = 20, alpha: float = 0.7, mmr_lambda: float = 0.5
) -> list[tuple[int, float, float]]:
    """
    Build weighted user profile vector from all feedback, score entire catalog.
    Returns list of (movie_id, combined_score, cosine_score) sorted by MMR.
    """
    from scipy.sparse import csr_matrix
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim

    log = session["swipe_log"]
    if not log:
        return []

    current_recommender = _get_recommender()
    feature_matrix = current_recommender.feature_matrix
    cosine_sim = current_recommender.cosine_sim
    if feature_matrix is None or cosine_sim is None:
        raise RuntimeError("Recommender artifacts are not initialized")

    n_features = feature_matrix.shape[1]
    profile = csr_matrix((1, n_features))

    for entry in log:
        idx = _movie_id_to_idx(entry["movie_id"])
        if idx is None:
            continue
        weight = FEEDBACK_WEIGHTS.get((entry["direction"], entry["source"]), 0.0)
        if weight == 0.0:
            continue
        profile = profile + feature_matrix[idx] * weight

    if profile.nnz == 0:
        return []

    norm = float(profile.multiply(profile).sum() ** 0.5)
    if norm > 0:
        profile = profile / norm

    cosine_scores = sk_cosine_sim(profile, feature_matrix).flatten()
    wr_norm = current_recommender.smd["wr_norm"].values
    combined = alpha * cosine_scores + (1 - alpha) * wr_norm
    seen_mask = current_recommender.smd["id"].isin(session["seen"]).values
    combined[seen_mask] = -np.inf

    disliked_indices = []
    for mid in session["disliked"]:
        idx = _movie_id_to_idx(mid)
        if idx is not None:
            disliked_indices.append(idx)

    if disliked_indices:
        max_sim_to_disliked = np.max(cosine_sim[:, disliked_indices], axis=1)
        penalty_mask = (max_sim_to_disliked > 0.7) & ~seen_mask
        combined[penalty_mask] *= 0.5

    n_candidates = n * 5
    top_indices = np.argsort(combined)[::-1][:n_candidates]
    top_indices = top_indices[combined[top_indices] > -np.inf]

    if len(top_indices) == 0:
        return []

    top_scores = combined[top_indices]

    selected = current_recommender._mmr_rerank(top_indices, top_scores, n, mmr_lambda)

    return [
        (
            int(current_recommender.smd.iloc[idx]["id"]),
            float(score),
            float(cosine_scores[idx]),
        )
        for idx, score in selected
    ]


@app.post("/api/swipe")
def swipe(req: SwipeRequest):
    """Record a user swipe (like / dislike) on a movie."""
    global pending_movie_meta
    current_movies_df = _get_movies_df()

    movie_row = current_movies_df[current_movies_df["id"] == req.movie_id]
    genres = []
    popularity = 0.0
    if not movie_row.empty:
        r = movie_row.iloc[0]
        genres = r["genres"] if isinstance(r["genres"], list) else []
        popularity = float(r["popularity"]) if pd.notna(r["popularity"]) else 0.0

    log_entry = {
        "movie_id": req.movie_id,
        "direction": req.direction,
        "source": pending_movie_meta["source"],
        "similarity_score": pending_movie_meta["similarity_score"],
        "combined_score": pending_movie_meta["combined_score"],
        "popularity": popularity,
        "genres": genres,
        "timestamp": time.time(),
        "swipe_index": len(session["swipe_log"]),
    }
    session["swipe_log"].append(log_entry)

    if req.direction == "right":
        session["liked"].append(req.movie_id)
    else:
        session["disliked"].append(req.movie_id)
    session["seen"].add(req.movie_id)

    pending_movie_meta = {
        "source": "random",
        "similarity_score": None,
        "combined_score": None,
    }
    return {"status": "ok"}


@app.post("/api/rec-feedback")
def rec_feedback(req: RecFeedbackRequest):
    """Record feedback on a recommendation list item."""
    movie_id = req.movie_id
    current_movies_df = _get_movies_df()

    if movie_id in session["seen"]:
        recs = _profile_recommendations(n=20)
        result = []
        for mid, combined_score, cosine_score in recs:
            row = current_movies_df[current_movies_df["id"] == mid]
            if not row.empty:
                d = movie_to_dict(row.iloc[0])
                d["similarity_score"] = round(cosine_score, 4)
                d["combined_score"] = round(combined_score, 4)
                result.append(d)
        return {
            "status": "duplicate",
            "recommendations": result,
            "liked_count": len(session["liked"]),
        }

    movie_row = current_movies_df[current_movies_df["id"] == movie_id]
    genres = []
    popularity = 0.0
    if not movie_row.empty:
        r = movie_row.iloc[0]
        genres = r["genres"] if isinstance(r["genres"], list) else []
        popularity = float(r["popularity"]) if pd.notna(r["popularity"]) else 0.0

    log_entry = {
        "movie_id": movie_id,
        "direction": req.direction,
        "source": "rec_list",
        "similarity_score": None,
        "combined_score": None,
        "popularity": popularity,
        "genres": genres,
        "timestamp": time.time(),
        "swipe_index": len(session["swipe_log"]),
    }
    session["swipe_log"].append(log_entry)

    if req.direction == "right":
        session["liked"].append(movie_id)
    else:
        session["disliked"].append(movie_id)
    session["seen"].add(movie_id)

    recs = _profile_recommendations(n=20)
    result = []
    for mid, combined_score, cosine_score in recs:
        row = current_movies_df[current_movies_df["id"] == mid]
        if not row.empty:
            d = movie_to_dict(row.iloc[0])
            d["similarity_score"] = round(cosine_score, 4)
            d["combined_score"] = round(combined_score, 4)
            result.append(d)

    return {
        "status": "ok",
        "recommendations": result,
        "liked_count": len(session["liked"]),
    }


@app.get("/api/movie")
def get_next_movie():
    """Return the next movie card the user should see."""
    global pending_movie_meta
    current_movies_df = _get_movies_df()
    seen = session["seen"]
    liked_count = len(session["liked"])

    if liked_count >= 3:
        if random.random() < EXPLORATION_RATE:
            unseen = current_movies_df[~current_movies_df["id"].isin(seen)]
            if not unseen.empty:
                weights = unseen["popularity"].clip(lower=0.1)
                weights = weights / weights.sum()
                chosen = unseen.sample(n=1, weights=weights)
                pending_movie_meta = {
                    "source": "random",
                    "similarity_score": None,
                    "combined_score": None,
                }
                return movie_to_dict(chosen.iloc[0], source="random")

        sorted_recs = _profile_recommendations(n=30)
        for mid, combined_score, cosine_score in sorted_recs:
            movie_row = current_movies_df[current_movies_df["id"] == mid]
            if not movie_row.empty:
                pending_movie_meta = {
                    "source": "model",
                    "similarity_score": round(cosine_score, 4),
                    "combined_score": round(combined_score, 4),
                }
                return movie_to_dict(movie_row.iloc[0], source="model")

    unseen = current_movies_df[~current_movies_df["id"].isin(seen)]
    if not unseen.empty:
        weights = unseen["popularity"].clip(lower=0.1)
        weights = weights / weights.sum()
        chosen = unseen.sample(n=1, weights=weights)
        pending_movie_meta = {
            "source": "random",
            "similarity_score": None,
            "combined_score": None,
        }
        return movie_to_dict(chosen.iloc[0], source="random")

    return {"error": "No more movies to show!"}


@app.get("/api/recommendations")
def get_recommendations():
    """Return current recommendations based on user profile vector."""
    if not session["liked"]:
        return {"recommendations": [], "liked_count": 0}

    current_movies_df = _get_movies_df()
    recs = _profile_recommendations(n=20)
    result = []
    for mid, combined_score, cosine_score in recs:
        movie_row = current_movies_df[current_movies_df["id"] == mid]
        if not movie_row.empty:
            d = movie_to_dict(movie_row.iloc[0])
            d["similarity_score"] = round(cosine_score, 4)
            d["combined_score"] = round(combined_score, 4)
            result.append(d)

    return {"recommendations": result, "liked_count": len(session["liked"])}


@app.get("/api/stats")
def get_stats():
    """Return session statistics."""
    rec_count = 0
    if session["liked"]:
        rec_count = len(_profile_recommendations(n=20))

    return {
        "liked": len(session["liked"]),
        "disliked": len(session["disliked"]),
        "recommendations_count": rec_count,
    }


@app.get("/api/reset")
def reset_session():
    """Clear all session state."""
    _reset_session_state()
    return {"status": "reset"}


@app.get("/api/models")
def list_models():
    """List available models and which is active."""
    return {
        "models": [
            {
                "mode": mode,
                "label": MODE_LABELS.get(mode, mode),
                "active": mode == active_mode,
            }
            for mode in sorted(models)
        ],
        "active": active_mode,
    }


class SwitchModelRequest(BaseModel):
    mode: str


@app.post("/api/models/switch")
def switch_model(req: SwitchModelRequest):
    """Switch the active recommender model."""
    global recommender, active_mode, movies_df
    if req.mode not in models:
        available = sorted(models)
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.mode}' not available. Available: {available}",
        )

    active_mode = req.mode
    recommender = models[active_mode]
    movies_df = recommender.smd.copy()
    _reset_session_state()
    return {
        "status": "ok",
        "active": active_mode,
        "label": MODE_LABELS.get(active_mode, active_mode),
    }


@app.get("/api/analytics/session")
def get_session_analytics():
    """Compute realtime session metrics from swipe log."""
    log = session["swipe_log"]
    total_swipes = len(log)

    if total_swipes == 0:
        return {
            "total_swipes": 0,
            "like_rate": 0.0,
            "source_breakdown": {},
            "model_like_rate": 0.0,
            "random_like_rate": 0.0,
            "rec_list_like_rate": 0.0,
            "model_lift": None,
            "hit_rate": 0.0,
        }

    total_likes = sum(1 for e in log if e["direction"] == "right")
    like_rate = total_likes / total_swipes

    source_breakdown = {}
    for source in ("random", "model", "rec_list"):
        entries = [e for e in log if e["source"] == source]
        count = len(entries)
        likes = sum(1 for e in entries if e["direction"] == "right")
        lr = likes / count if count > 0 else 0.0
        source_breakdown[source] = {
            "swipes": count,
            "likes": likes,
            "like_rate": round(lr, 4),
        }

    random_lr = source_breakdown["random"]["like_rate"]
    model_lr = source_breakdown["model"]["like_rate"]
    rec_list_lr = source_breakdown["rec_list"]["like_rate"]

    model_lift = round(model_lr / random_lr, 2) if random_lr > 0 else None
    hit_rate = model_lr

    return {
        "total_swipes": total_swipes,
        "like_rate": round(like_rate, 4),
        "source_breakdown": source_breakdown,
        "model_like_rate": round(model_lr, 4),
        "random_like_rate": round(random_lr, 4),
        "rec_list_like_rate": round(rec_list_lr, 4),
        "model_lift": model_lift,
        "hit_rate": round(hit_rate, 4),
    }


@app.get("/api/analytics/offline")
def get_offline_analytics():
    """Run offline evaluation metrics (cached after first call)."""
    global offline_cache
    if offline_cache is not None:
        return offline_cache

    if recommender is None:
        return {}

    from src.models.evaluate_model import evaluate_all

    smd = recommender.smd
    test = smd[smd["vote_count"] >= smd["vote_count"].quantile(0.6)].sample(
        n=min(100, len(smd)), random_state=42
    )

    results = evaluate_all(recommender, test, k=5)
    serialized = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serialized[k] = {sk: round(float(sv), 4) for sk, sv in v.items()}
        else:
            serialized[k] = round(float(v), 4)

    offline_cache = serialized
    return offline_cache


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_frontend():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Movie Swipe Recommender API. Frontend not yet deployed."}
