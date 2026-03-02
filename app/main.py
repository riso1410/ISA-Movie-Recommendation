"""
FastAPI backend for Movie Swipe Recommender.
Tinder-style movie discovery with content-based recommendations.
"""

import time
import threading
import uuid
from datetime import datetime

import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ast import literal_eval
import pickle
import random
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
session: dict = {"liked": [], "disliked": [], "seen": set(), "swipe_log": []}
pending_movie_meta: dict = {"source": "random", "similarity_score": None, "combined_score": None}
offline_cache: dict | None = None

# Feedback weight constants: (direction, source) → weight
FEEDBACK_WEIGHTS: dict[tuple[str, str], float] = {
    ("right", "random"):    1.0,
    ("left",  "random"):   -0.3,
    ("right", "model"):     2.5,
    ("left",  "model"):    -1.2,
    ("right", "rec_list"):  3.0,
    ("left",  "rec_list"): -1.5,
}
EXPLORATION_RATE = 0.15

# Background task infrastructure
_tasks: dict[str, dict] = {}
_task_lock = threading.Lock()
_active_task_id: str | None = None
_cancel_flags: dict[str, threading.Event] = {}


def _create_task(task_type: str) -> str:
    global _active_task_id
    with _task_lock:
        if _active_task_id is not None:
            raise RuntimeError(f"Task {_active_task_id} already running")
        # Prune old completed/error tasks, keep last 20
        done = [tid for tid, t in _tasks.items() if t["status"] in ("completed", "error")]
        if len(done) > 20:
            for tid in sorted(done, key=lambda t: _tasks[t].get("started_at", ""))[:len(done) - 20]:
                del _tasks[tid]
        task_id = str(uuid.uuid4())[:8]
        _tasks[task_id] = {
            "type": task_type, "status": "running", "progress": 0.0,
            "progress_msg": "Starting...", "result": None, "error": None,
            "started_at": datetime.now().isoformat(), "finished_at": None,
        }
        _active_task_id = task_id
        _cancel_flags[task_id] = threading.Event()
        return task_id


def _update_progress(task_id: str, progress: float, msg: str) -> None:
    with _task_lock:
        if task_id in _tasks:
            _tasks[task_id]["progress"] = round(progress, 3)
            _tasks[task_id]["progress_msg"] = msg


def _finish_task(task_id: str, result: dict | None = None, error: str | None = None) -> None:
    global _active_task_id
    with _task_lock:
        if task_id in _tasks:
            _tasks[task_id]["status"] = "error" if error else "completed"
            _tasks[task_id]["result"] = result
            _tasks[task_id]["error"] = error
            _tasks[task_id]["progress"] = 1.0 if not error else _tasks[task_id]["progress"]
            _tasks[task_id]["finished_at"] = datetime.now().isoformat()
        if _active_task_id == task_id:
            _active_task_id = None
        _cancel_flags.pop(task_id, None)


def _hot_swap_model(new_rec: ContentBasedRecommender) -> None:
    global recommender, movies_df, offline_cache
    # Preserve poster URLs from current data
    if movies_df is not None and "poster_url" in movies_df.columns:
        poster_map = movies_df.drop_duplicates(subset="id").set_index("id")["poster_url"]
        new_rec.smd["poster_url"] = new_rec.smd["id"].map(poster_map).fillna("")
    elif "poster_url" not in new_rec.smd.columns:
        new_rec.smd["poster_url"] = ""
    # Fallback: use poster_path for movies still missing poster_url
    if "poster_path" in new_rec.smd.columns:
        missing = new_rec.smd["poster_url"] == ""
        paths = new_rec.smd.loc[missing, "poster_path"].fillna("")
        new_rec.smd.loc[missing, "poster_url"] = paths.apply(
            lambda p: f"https://image.tmdb.org/t/p/w500{p}" if p else ""
        )
    movies_df = new_rec.smd.copy()
    recommender = new_rec
    offline_cache = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class SwipeRequest(BaseModel):
    movie_id: int
    direction: str  # "right" (like) or "left" (dislike)


class WeightsRequest(BaseModel):
    weights: dict[str, float]


class RecFeedbackRequest(BaseModel):
    movie_id: int
    direction: str  # "right" or "left"


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
        "vote_count": int(row["vote_count"])
        if pd.notna(row["vote_count"])
        else 0,
        "poster_url": str(row.get("poster_url", "")) if pd.notna(row.get("poster_url", "")) else "",
    }
    if source is not None:
        d["source"] = source
    return d


def _get_liked_titles() -> list[str]:
    """Return titles corresponding to the liked movie ids."""
    if not session["liked"]:
        return []
    liked_rows = movies_df[movies_df["id"].isin(session["liked"])]
    return liked_rows["title"].tolist()


def _movie_id_to_idx(movie_id: int) -> int | None:
    """Map movie ID to recommender.smd row index."""
    matches = recommender.smd.index[recommender.smd["id"] == movie_id]
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

    n_features = recommender.feature_matrix.shape[1]
    profile = csr_matrix((1, n_features))

    for entry in log:
        idx = _movie_id_to_idx(entry["movie_id"])
        if idx is None:
            continue
        weight = FEEDBACK_WEIGHTS.get((entry["direction"], entry["source"]), 0.0)
        if weight == 0.0:
            continue
        profile = profile + recommender.feature_matrix[idx] * weight

    if profile.nnz == 0:
        return []

    # L2 normalize
    norm = float(profile.multiply(profile).sum() ** 0.5)
    if norm > 0:
        profile = profile / norm

    # Cosine similarity: profile vs all movies
    cosine_scores = sk_cosine_sim(profile, recommender.feature_matrix).flatten()

    # Combined score
    wr_norm = recommender.smd["wr_norm"].values
    combined = alpha * cosine_scores + (1 - alpha) * wr_norm

    # Hard exclude seen movies
    seen_mask = recommender.smd["id"].isin(session["seen"]).values
    combined[seen_mask] = -np.inf

    # Soft penalty: penalize movies similar to disliked ones
    disliked_indices = []
    for mid in session["disliked"]:
        idx = _movie_id_to_idx(mid)
        if idx is not None:
            disliked_indices.append(idx)

    if disliked_indices:
        max_sim_to_disliked = np.max(
            recommender.cosine_sim[:, disliked_indices], axis=1
        )
        penalty_mask = (max_sim_to_disliked > 0.7) & ~seen_mask
        combined[penalty_mask] *= 0.5

    # Get top candidates for MMR
    n_candidates = n * 5
    top_indices = np.argsort(combined)[::-1][:n_candidates]
    top_indices = top_indices[combined[top_indices] > -np.inf]

    if len(top_indices) == 0:
        return []

    top_scores = combined[top_indices]

    # MMR rerank
    selected = recommender._mmr_rerank(top_indices, top_scores, n, mmr_lambda)

    return [
        (int(recommender.smd.iloc[idx]["id"]), float(score), float(cosine_scores[idx]))
        for idx, score in selected
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/swipe")
def swipe(req: SwipeRequest):
    """Record a user swipe (like / dislike) on a movie."""
    global pending_movie_meta

    # Build swipe log entry
    movie_row = movies_df[movies_df["id"] == req.movie_id]
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

    # Reset pending meta
    pending_movie_meta = {"source": "random", "similarity_score": None, "combined_score": None}
    return {"status": "ok"}


@app.post("/api/rec-feedback")
def rec_feedback(req: RecFeedbackRequest):
    """Record feedback on a recommendation list item."""
    movie_id = req.movie_id

    # Guard: skip state mutation if already seen (stale UI)
    if movie_id in session["seen"]:
        recs = _profile_recommendations(n=20)
        result = []
        for mid, combined_score, cosine_score in recs:
            row = movies_df[movies_df["id"] == mid]
            if not row.empty:
                d = movie_to_dict(row.iloc[0])
                d["similarity_score"] = round(cosine_score, 4)
                d["combined_score"] = round(combined_score, 4)
                result.append(d)
        return {"status": "duplicate", "recommendations": result, "liked_count": len(session["liked"])}

    # Build log entry
    movie_row = movies_df[movies_df["id"] == movie_id]
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

    # Return refreshed recs
    recs = _profile_recommendations(n=20)
    result = []
    for mid, combined_score, cosine_score in recs:
        row = movies_df[movies_df["id"] == mid]
        if not row.empty:
            d = movie_to_dict(row.iloc[0])
            d["similarity_score"] = round(cosine_score, 4)
            d["combined_score"] = round(combined_score, 4)
            result.append(d)

    return {"status": "ok", "recommendations": result, "liked_count": len(session["liked"])}


@app.get("/api/movie")
def get_next_movie():
    """Return the next movie card the user should see."""
    global pending_movie_meta
    seen = session["seen"]
    liked_count = len(session["liked"])

    # After 3+ likes, use profile-based recs with exploration
    if liked_count >= 3:
        # 15% exploration: random unseen movie even in model phase
        if random.random() < EXPLORATION_RATE:
            unseen = movies_df[~movies_df["id"].isin(seen)]
            if not unseen.empty:
                weights = unseen["popularity"].clip(lower=0.1)
                weights = weights / weights.sum()
                chosen = unseen.sample(n=1, weights=weights)
                pending_movie_meta = {"source": "random", "similarity_score": None, "combined_score": None}
                return movie_to_dict(chosen.iloc[0], source="random")

        # Model-based pick from profile vector
        sorted_recs = _profile_recommendations(n=30)
        for mid, combined_score, cosine_score in sorted_recs:
            movie_row = movies_df[movies_df["id"] == mid]
            if not movie_row.empty:
                pending_movie_meta = {
                    "source": "model",
                    "similarity_score": round(cosine_score, 4),
                    "combined_score": round(combined_score, 4),
                }
                return movie_to_dict(movie_row.iloc[0], source="model")

    # Fallback: random unseen movie (weighted by popularity)
    unseen = movies_df[~movies_df["id"].isin(seen)]
    if not unseen.empty:
        weights = unseen["popularity"].clip(lower=0.1)
        weights = weights / weights.sum()
        chosen = unseen.sample(n=1, weights=weights)
        pending_movie_meta = {"source": "random", "similarity_score": None, "combined_score": None}
        return movie_to_dict(chosen.iloc[0], source="random")

    return {"error": "No more movies to show!"}


@app.get("/api/recommendations")
def get_recommendations():
    """Return current recommendations based on user profile vector."""
    if not session["liked"]:
        return {"recommendations": [], "liked_count": 0}

    recs = _profile_recommendations(n=20)
    result = []
    for mid, combined_score, cosine_score in recs:
        movie_row = movies_df[movies_df["id"] == mid]
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
    global pending_movie_meta, offline_cache
    session["liked"] = []
    session["disliked"] = []
    session["seen"] = set()
    session["swipe_log"] = []
    pending_movie_meta = {"source": "random", "similarity_score": None, "combined_score": None}
    offline_cache = None
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------
@app.get("/api/analytics")
def get_analytics():
    """Session analytics computed from swipe_log."""
    log = session["swipe_log"]
    total = len(log)
    if total == 0:
        return {
            "total_swipes": 0, "like_rates": {}, "confusion": {},
            "swipe_buckets": [], "genre_distribution": {},
            "popularity": {}, "score_distributions": {},
            "calibration": [], "swipe_log": [],
        }

    likes = [e for e in log if e["direction"] == "right"]
    dislikes = [e for e in log if e["direction"] == "left"]
    random_swipes = [e for e in log if e["source"] == "random"]
    model_swipes = [e for e in log if e["source"] == "model"]
    rec_list_swipes = [e for e in log if e["source"] == "rec_list"]
    random_likes = [e for e in random_swipes if e["direction"] == "right"]
    model_likes = [e for e in model_swipes if e["direction"] == "right"]
    rec_list_likes = [e for e in rec_list_swipes if e["direction"] == "right"]

    lr_overall = len(likes) / total if total else 0
    lr_random = len(random_likes) / len(random_swipes) if random_swipes else 0
    lr_model = len(model_likes) / len(model_swipes) if model_swipes else 0
    lr_rec_list = len(rec_list_likes) / len(rec_list_swipes) if rec_list_swipes else 0
    model_lift = (lr_model / lr_random) if lr_random > 0 else 0

    # Source breakdown
    source_breakdown = {
        "random": {"total": len(random_swipes), "likes": len(random_likes)},
        "model": {"total": len(model_swipes), "likes": len(model_likes)},
        "rec_list": {"total": len(rec_list_swipes), "likes": len(rec_list_likes)},
    }

    # Confusion matrix: TP=liked+model/rec_list, FP=disliked+model/rec_list, FN=liked+random, TN=disliked+random
    model_and_rec = model_swipes + rec_list_swipes
    tp = len([e for e in model_and_rec if e["direction"] == "right"])
    fp = len([e for e in model_and_rec if e["direction"] == "left"])
    fn = len([e for e in random_swipes if e["direction"] == "right"])
    tn = len([e for e in random_swipes if e["direction"] == "left"])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Swipe buckets (groups of 5)
    bucket_size = 5
    buckets = []
    for i in range(0, total, bucket_size):
        chunk = log[i:i + bucket_size]
        chunk_likes = sum(1 for e in chunk if e["direction"] == "right")
        chunk_model = sum(1 for e in chunk if e["source"] in ("model", "rec_list"))
        buckets.append({
            "bucket": i // bucket_size,
            "start": i,
            "end": min(i + bucket_size, total),
            "like_rate": chunk_likes / len(chunk),
            "model_count": chunk_model,
            "random_count": len(chunk) - chunk_model,
        })

    # Genre distribution
    genre_dist: dict[str, dict] = {}
    for e in log:
        for g in e["genres"]:
            if g not in genre_dist:
                genre_dist[g] = {"liked": 0, "disliked": 0}
            if e["direction"] == "right":
                genre_dist[g]["liked"] += 1
            else:
                genre_dist[g]["disliked"] += 1

    # Popularity
    liked_pop = [e["popularity"] for e in likes] if likes else [0]
    disliked_pop = [e["popularity"] for e in dislikes] if dislikes else [0]
    catalog_pop = float(movies_df["popularity"].mean()) if movies_df is not None else 0

    # Score distributions (model recs only)
    liked_scores = [e["similarity_score"] for e in model_likes if e["similarity_score"] is not None]
    disliked_model = [e for e in model_swipes if e["direction"] == "left"]
    disliked_scores = [e["similarity_score"] for e in disliked_model if e["similarity_score"] is not None]

    # Calibration: score bins with like rate
    calibration = []
    model_with_scores = [e for e in model_swipes if e["similarity_score"] is not None]
    if model_with_scores:
        for bin_start in np.arange(0, 1.0, 0.1):
            bin_end = bin_start + 0.1
            in_bin = [e for e in model_with_scores if bin_start <= e["similarity_score"] < bin_end]
            if in_bin:
                bin_likes = sum(1 for e in in_bin if e["direction"] == "right")
                calibration.append({
                    "bin": f"{bin_start:.1f}-{bin_end:.1f}",
                    "like_rate": bin_likes / len(in_bin),
                    "count": len(in_bin),
                })

    return {
        "total_swipes": total,
        "like_rates": {
            "overall": round(lr_overall, 3),
            "random": round(lr_random, 3),
            "model": round(lr_model, 3),
            "rec_list": round(lr_rec_list, 3),
            "model_lift": round(model_lift, 2),
        },
        "source_breakdown": source_breakdown,
        "confusion": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        },
        "swipe_buckets": buckets,
        "genre_distribution": genre_dist,
        "popularity": {
            "liked_avg": round(float(np.mean(liked_pop)), 1),
            "disliked_avg": round(float(np.mean(disliked_pop)), 1),
            "catalog_avg": round(catalog_pop, 1),
        },
        "score_distributions": {
            "liked": liked_scores,
            "disliked": disliked_scores,
        },
        "calibration": calibration,
        "swipe_log": log,
    }


@app.get("/api/analytics/catalog")
def get_catalog_analytics():
    """Catalog-level analytics (genre distribution, popularity histogram)."""
    if movies_df is None:
        return {}

    # Genre distribution
    genre_counts: dict[str, int] = {}
    for genres in movies_df["genres"]:
        if isinstance(genres, list):
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + 1

    # Popularity histogram
    pop = pd.to_numeric(movies_df["popularity"], errors="coerce").fillna(0.0)
    p95 = float(np.percentile(pop, 95))
    clipped = pop.clip(upper=p95)
    hist_values, hist_edges = np.histogram(clipped, bins=20)

    return {
        "total_movies": len(movies_df),
        "genre_distribution": genre_counts,
        "popularity_histogram": {
            "counts": hist_values.tolist(),
            "edges": [round(float(e), 1) for e in hist_edges],
        },
    }


@app.get("/api/analytics/model")
def get_model_analytics():
    """Model training stats."""
    if recommender is None:
        return {}

    stats = recommender.get_model_stats()

    # Add similarity distribution histogram
    if recommender.cosine_sim is not None:
        upper = recommender.cosine_sim[np.triu_indices(len(recommender.cosine_sim), k=1)]
        sample = np.random.choice(upper, size=min(100000, len(upper)), replace=False)
        p95 = float(np.percentile(sample, 95))
        hist_values, hist_edges = np.histogram(sample, bins=20, range=(0, p95))
        stats["similarity_histogram"] = {
            "counts": hist_values.tolist(),
            "edges": [round(float(e), 4) for e in hist_edges],
        }

    return stats


@app.get("/api/analytics/offline")
def get_offline_analytics():
    """Run offline evaluation metrics (cached after first call)."""
    global offline_cache
    if offline_cache is not None:
        return offline_cache

    if recommender is None:
        return {}

    from src.models.evaluate_model import evaluate_all

    # Sample test movies
    smd = recommender.smd
    test = smd[smd["vote_count"] >= smd["vote_count"].quantile(0.6)].sample(
        n=min(100, len(smd)), random_state=42
    )

    results = evaluate_all(recommender, test, k=10)

    # Convert numpy types for JSON serialization
    serialized = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serialized[k] = {sk: round(float(sv), 4) for sk, sv in v.items()}
        else:
            serialized[k] = round(float(v), 4)

    offline_cache = serialized
    return offline_cache


# ---------------------------------------------------------------------------
# Task management endpoints
# ---------------------------------------------------------------------------
@app.get("/api/tasks")
def list_tasks():
    with _task_lock:
        return {"tasks": dict(_tasks), "active_task_id": _active_task_id}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    with _task_lock:
        if task_id not in _tasks:
            raise HTTPException(404, "Task not found")
        return _tasks[task_id]


@app.post("/api/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    with _task_lock:
        if task_id not in _cancel_flags:
            raise HTTPException(404, "Task not found or not cancellable")
        _cancel_flags[task_id].set()
    return {"status": "cancel_requested"}


# ---------------------------------------------------------------------------
# Operation endpoints — heavy (background thread)
# ---------------------------------------------------------------------------
@app.post("/api/operations/retrain")
def op_retrain():
    try:
        task_id = _create_task("retrain")
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    def _run():
        try:
            from src.data.make_dataset import make_dataset

            _update_progress(task_id, 0.1, "Loading raw data...")
            raw_dir = str(PROJECT_ROOT / "data" / "raw")
            smd = make_dataset(raw_dir)

            _update_progress(task_id, 0.3, "Building features...")
            smd = build_features(smd)

            _update_progress(task_id, 0.5, "Fitting recommender...")
            from src.models.predict_model import build_recommender
            new_rec = build_recommender(smd)

            _update_progress(task_id, 0.8, "Saving model...")
            model_dir = PROJECT_ROOT / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "recommender.pkl", "wb") as f:
                pickle.dump(new_rec, f)

            _update_progress(task_id, 0.9, "Hot-swapping model...")
            _hot_swap_model(new_rec)

            _finish_task(task_id, {"message": f"Retrained with {len(smd)} movies"})
        except Exception as e:
            _finish_task(task_id, error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id}


@app.post("/api/operations/evaluate")
def op_evaluate():
    try:
        task_id = _create_task("evaluate")
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    def _run():
        try:
            from src.models.evaluate_model import evaluate_all
            global offline_cache

            _update_progress(task_id, 0.1, "Selecting test movies...")
            smd = recommender.smd
            test = smd[smd["vote_count"] >= smd["vote_count"].quantile(0.6)].sample(
                n=min(100, len(smd)), random_state=42
            )

            _update_progress(task_id, 0.2, "Running evaluation metrics...")
            results = evaluate_all(recommender, test, k=10)

            _update_progress(task_id, 0.9, "Serializing results...")
            serialized = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    serialized[k] = {sk: round(float(sv), 4) for sk, sv in v.items()}
                else:
                    serialized[k] = round(float(v), 4)

            offline_cache = serialized
            _finish_task(task_id, serialized)
        except Exception as e:
            _finish_task(task_id, error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id}


@app.post("/api/operations/gridsearch")
def op_gridsearch():
    try:
        task_id = _create_task("gridsearch")
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    def _run():
        try:
            from itertools import product as iter_product
            from src.models.predict_model import ContentBasedRecommender as CBR, DEFAULT_WEIGHTS
            from src.models.evaluate_model import precision_at_k, ndcg_at_k

            _update_progress(task_id, 0.05, "Preparing data...")
            smd = recommender.smd.copy()
            smd = build_features(smd)

            test = smd[smd["vote_count"] >= smd["vote_count"].quantile(0.6)].sample(
                n=min(100, len(smd)), random_state=42
            )

            weight_ranges = {
                "genres": [0.5, 1.0, 1.5, 2.0],
                "keywords": [0.5, 1.0, 1.5, 2.0],
                "director": [1.0, 1.5, 2.0, 2.5],
                "collection": [0.5, 1.0, 1.5, 2.0],
            }
            fields = list(weight_ranges.keys())
            value_lists = [weight_ranges[f] for f in fields]
            combos = list(iter_product(*value_lists))
            total = len(combos)

            cancel_flag = _cancel_flags.get(task_id)
            best_score = -1.0
            best_weights = None
            results = []

            for i, combo in enumerate(combos):
                if cancel_flag and cancel_flag.is_set():
                    _finish_task(task_id, result={
                        "cancelled": True, "completed_combos": i,
                        "total_combos": total, "best_weights": best_weights,
                        "best_score": round(best_score, 4),
                        "top_10": sorted(results, key=lambda x: x["combined"], reverse=True)[:10],
                    })
                    return

                weights = DEFAULT_WEIGHTS.copy()
                for field, val in zip(fields, combo):
                    weights[field] = val

                rec = CBR(smd, weights=weights)
                rec.fit()

                p_at_k = precision_at_k(rec, test, 10)
                n_at_k = ndcg_at_k(rec, test, 10)
                combined = 0.5 * p_at_k + 0.5 * n_at_k

                entry = {
                    "weights": dict(zip(fields, combo)),
                    "precision_at_k": round(p_at_k, 4),
                    "ndcg_at_k": round(n_at_k, 4),
                    "combined": round(combined, 4),
                }
                results.append(entry)

                if combined > best_score:
                    best_score = combined
                    best_weights = weights.copy()

                if (i + 1) % 5 == 0:
                    _update_progress(task_id, (i + 1) / total, f"{i+1}/{total} combos evaluated")

            _finish_task(task_id, {
                "best_weights": best_weights, "best_score": round(best_score, 4),
                "total_combos": total,
                "top_10": sorted(results, key=lambda x: x["combined"], reverse=True)[:10],
            })
        except Exception as e:
            _finish_task(task_id, error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id}


@app.post("/api/operations/fetch-posters")
def op_fetch_posters():
    try:
        task_id = _create_task("fetch_posters")
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    def _run():
        try:
            import random as _rnd
            from src.data.fetch_posters import fetch_one, load_cache, save_cache

            cache_path = PROJECT_ROOT / "data" / "processed" / "poster_cache.csv"
            processed_path = PROJECT_ROOT / "data" / "processed" / "movies_processed.csv"

            _update_progress(task_id, 0.01, "Loading cache...")
            cache = load_cache(cache_path)

            df = pd.read_csv(processed_path)
            all_ids = df["id"].astype(int).tolist()
            remaining = [mid for mid in all_ids if mid not in cache]

            if not remaining:
                _finish_task(task_id, {"message": "All posters already cached", "total": len(all_ids), "cached": len(cache)})
                return

            cancel_flag = _cancel_flags.get(task_id)
            fetched = 0
            found = 0
            _rnd.shuffle(remaining)

            for i, mid in enumerate(remaining):
                if cancel_flag and cancel_flag.is_set():
                    save_cache(cache, cache_path)
                    _finish_task(task_id, {
                        "cancelled": True, "fetched": fetched, "found": found,
                        "remaining": len(remaining) - i, "total_cached": len(cache),
                    })
                    return

                result = fetch_one(mid)
                if result is None:
                    # Rate-limited: wait then retry, but check cancel first
                    if cancel_flag and cancel_flag.wait(timeout=60):
                        save_cache(cache, cache_path)
                        _finish_task(task_id, {
                            "cancelled": True, "fetched": fetched, "found": found,
                            "remaining": len(remaining) - i, "total_cached": len(cache),
                        })
                        return
                    result = fetch_one(mid)

                if result:
                    cache[mid] = result
                    found += 1
                fetched += 1

                if fetched % 20 == 0:
                    save_cache(cache, cache_path)

                pct = (i + 1) / len(remaining)
                _update_progress(task_id, pct, f"{i+1}/{len(remaining)} — {found} posters found")

                time.sleep(_rnd.uniform(2.0, 4.0))

            save_cache(cache, cache_path)
            # Write poster_url to processed CSV
            df["poster_url"] = df["id"].map(cache).fillna("")
            df.to_csv(processed_path, index=False)

            _finish_task(task_id, {"fetched": fetched, "found": found, "total_cached": len(cache)})
        except Exception as e:
            _finish_task(task_id, error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id}


@app.post("/api/operations/apply-weights")
def op_apply_weights(req: WeightsRequest):
    try:
        task_id = _create_task("apply_weights")
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    weights = req.weights

    def _run():
        try:
            from src.models.predict_model import build_recommender

            _update_progress(task_id, 0.1, "Building features...")
            smd = recommender.smd.copy()
            smd = build_features(smd)

            _update_progress(task_id, 0.4, "Fitting recommender with new weights...")
            new_rec = build_recommender(smd, weights=weights)

            _update_progress(task_id, 0.7, "Saving model...")
            model_dir = PROJECT_ROOT / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "recommender.pkl", "wb") as f:
                pickle.dump(new_rec, f)

            _update_progress(task_id, 0.9, "Hot-swapping model...")
            _hot_swap_model(new_rec)

            _finish_task(task_id, {"message": "Weights applied", "weights": weights})
        except Exception as e:
            _finish_task(task_id, error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id}


# ---------------------------------------------------------------------------
# Operation endpoints — quick (synchronous)
# ---------------------------------------------------------------------------
@app.post("/api/operations/reload-model")
def op_reload_model():
    global recommender, movies_df, offline_cache
    model_path = PROJECT_ROOT / "models" / "recommender.pkl"
    if not model_path.exists():
        raise HTTPException(404, "No model file found")

    with open(model_path, "rb") as f:
        new_rec = pickle.load(f)

    new_rec.smd = new_rec.smd.reset_index(drop=True)
    for col in ["vote_average", "vote_count", "popularity"]:
        new_rec.smd[col] = pd.to_numeric(new_rec.smd[col], errors="coerce").fillna(0.0)

    _hot_swap_model(new_rec)
    return {"status": "ok", "movies": len(movies_df)}


@app.post("/api/operations/reload-data")
def op_reload_data():
    global movies_df
    processed_path = PROJECT_ROOT / "data" / "processed" / "movies_processed.csv"
    if not processed_path.exists():
        raise HTTPException(404, "No processed data found")

    processed = pd.read_csv(processed_path)
    for col in ["genres", "keywords", "cast"]:
        processed[col] = processed[col].apply(
            lambda x: literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
        )
    for col in ["decade", "language", "collection", "poster_url"]:
        if col in processed.columns:
            processed[col] = processed[col].fillna("")

    # Update poster URLs in recommender
    if recommender is not None:
        csv_poster = processed.drop_duplicates(subset="id").set_index("id")["poster_url"]
        recommender.smd["poster_url"] = recommender.smd["id"].map(csv_poster).fillna("")
        movies_df = recommender.smd.copy()
    return {"status": "ok", "movies": len(processed)}


@app.get("/api/model/weights")
def get_model_weights():
    if recommender is None:
        raise HTTPException(503, "No model loaded")
    return {"weights": recommender.weights}


# ---------------------------------------------------------------------------
# Static file serving (frontend)
# ---------------------------------------------------------------------------
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/dashboard")
def serve_dashboard():
    dashboard = static_dir / "dashboard.html"
    if dashboard.exists():
        return FileResponse(str(dashboard))
    return {"error": "Dashboard not found"}


@app.get("/")
def serve_frontend():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Movie Swipe Recommender API. Frontend not yet deployed."}
