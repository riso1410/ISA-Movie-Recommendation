"""
Evaluation metrics for content-based recommender.

Two evaluation modes:
  1. Ratings-based (primary): Leave-N-out with real MovieLens user ratings as ground truth.
     Metrics: Precision@K, Recall@K, NDCG@K, MAP@K, MRR, HR@K, Coverage, ILD, Novelty,
     Serendipity.
  2. Calibration (Steck 2018): Measures whether the genre/decade distribution of
     recommendations matches the user's preference distribution.
     Metrics: KL-divergence, Jensen-Shannon divergence, Calibration Score.
"""

import sys
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

from src.data.make_dataset import make_dataset, load_ratings
from src.features.build_features import build_features
from src.models.predict_model import ContentBasedRecommender, DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_tmdb_to_idx(recommender: ContentBasedRecommender) -> dict[int, int]:
    """Map TMDB movie ID -> index in recommender.smd."""
    mapping = {}
    for idx, movie_id in enumerate(recommender.smd["id"].values):
        try:
            tid = int(movie_id)
        except (ValueError, TypeError):
            continue
        if tid not in mapping:
            mapping[tid] = idx
    return mapping


def _prepare_user_profiles(
    recommender: ContentBasedRecommender,
    n_users: int = 200,
    min_liked: int = 10,
    like_threshold: float = 4.0,
    random_state: int = 42,
) -> list[dict]:
    """Build user profiles from MovieLens ratings with 70/30 train/test split.

    Returns list of dicts with keys: user_id, profile_indices, test_indices.
    """
    ratings = load_ratings()
    tmdb_to_idx = _build_tmdb_to_idx(recommender)

    # Map ratings to recommender indices
    ratings = ratings[ratings["tmdbId"].isin(tmdb_to_idx)]
    ratings["idx"] = ratings["tmdbId"].map(tmdb_to_idx)

    # Keep only liked movies
    liked = ratings[ratings["rating"] >= like_threshold]

    # Filter users with enough liked movies
    user_counts = liked.groupby("userId").size()
    qualifying_users = user_counts[user_counts >= min_liked].index.tolist()

    if not qualifying_users:
        print(f"  Warning: No users with >= {min_liked} liked movies. Trying min_liked=5...")
        min_liked = 5
        qualifying_users = user_counts[user_counts >= min_liked].index.tolist()

    rng = np.random.RandomState(random_state)
    if len(qualifying_users) > n_users:
        qualifying_users = rng.choice(qualifying_users, size=n_users, replace=False).tolist()

    profiles = []
    for uid in qualifying_users:
        user_liked = liked[liked["userId"] == uid]["idx"].values
        n_profile = max(1, int(len(user_liked) * 0.7))

        shuffled = rng.permutation(user_liked)
        profile_idx = shuffled[:n_profile].tolist()
        test_idx = shuffled[n_profile:].tolist()

        if not test_idx:
            continue

        profiles.append({
            "user_id": uid,
            "profile_indices": profile_idx,
            "test_indices": test_idx,
        })

    return profiles


# ---------------------------------------------------------------------------
# Ratings-based evaluation (primary)
# ---------------------------------------------------------------------------


def evaluate_with_ratings(
    recommender: ContentBasedRecommender,
    k_values: list[int] | None = None,
    n_users: int = 200,
    alpha: float = 0.7,
    mmr_lambda: float = 0.5,
) -> dict:
    """Primary evaluation: leave-N-out with real MovieLens ratings.

    For each user: build profile from 70% liked movies, test against held-out 30%.
    Returns metrics at each K value + system-level metrics.
    """
    if k_values is None:
        k_values = [5, 10]
    max_k = max(k_values)

    print("  Preparing user profiles...")
    profiles = _prepare_user_profiles(recommender, n_users=n_users)
    print(f"  Evaluating {len(profiles)} users at K={k_values}...")

    if not profiles:
        print("  ERROR: No qualifying user profiles found.")
        return {}

    # Per-user metrics storage
    metrics_by_k: dict[int, dict[str, list]] = {}
    for k in k_values:
        metrics_by_k[k] = {
            "precision": [], "recall": [], "ndcg": [],
            "map": [], "mrr": [], "hit_rate": [],
        }

    # System-level tracking
    all_recommended = set()
    all_ild_scores = []
    all_novelty_scores = []
    all_serendipity_scores = []

    # Popularity baseline for serendipity
    pop = pd.to_numeric(recommender.smd["popularity"], errors="coerce").fillna(0.0)
    popular_baseline = set(np.argsort(pop.values)[::-1][:max_k].tolist())

    pop_max = pop.max()
    pop_norm = (pop / pop_max).values if pop_max > 0 else np.zeros(len(pop))

    cosine_sim = recommender.cosine_sim

    for profile in profiles:
        profile_idx = profile["profile_indices"]
        test_set = set(profile["test_indices"])

        rec_indices, rec_scores = recommender.recommend_from_profile(
            profile_idx, n=max_k, alpha=alpha, mmr_lambda=mmr_lambda,
        )

        all_recommended.update(rec_indices)

        # ILD for this user's list
        if len(rec_indices) >= 2 and cosine_sim is not None:
            sub_sim = cosine_sim[np.ix_(rec_indices, rec_indices)]
            n_items = len(rec_indices)
            upper_mask = np.triu_indices(n_items, k=1)
            mean_sim = sub_sim[upper_mask].mean()
            all_ild_scores.append(1.0 - mean_sim)

        # Novelty for this user's list
        for idx in rec_indices:
            p = float(pop_norm[idx])
            all_novelty_scores.append(-np.log2(max(p, 1e-10)))

        # Serendipity: relevant items NOT in popularity baseline
        relevant_recs = [idx for idx in rec_indices if idx in test_set]
        serendipitous = [idx for idx in relevant_recs if idx not in popular_baseline]
        if relevant_recs:
            all_serendipity_scores.append(len(serendipitous) / len(relevant_recs))

        # Per-K metrics
        for k in k_values:
            top_k = rec_indices[:k]
            top_k_set = set(top_k)

            hits = top_k_set & test_set
            n_relevant_total = len(test_set)

            # Precision@K
            metrics_by_k[k]["precision"].append(len(hits) / k)

            # Recall@K
            metrics_by_k[k]["recall"].append(len(hits) / n_relevant_total)

            # NDCG@K (binary relevance)
            relevances = [1.0 if idx in test_set else 0.0 for idx in top_k]
            dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevances))
            ideal = sorted(relevances, reverse=True)
            idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
            metrics_by_k[k]["ndcg"].append(dcg / idcg if idcg > 0 else 0.0)

            # MAP@K (fixed denominator: min(k, total_relevant))
            n_hits = 0
            sum_prec = 0.0
            for rank, idx in enumerate(top_k, 1):
                if idx in test_set:
                    n_hits += 1
                    sum_prec += n_hits / rank
            denom = min(k, n_relevant_total)
            metrics_by_k[k]["map"].append(sum_prec / denom if denom > 0 else 0.0)

            # MRR
            rr = 0.0
            for rank, idx in enumerate(top_k, 1):
                if idx in test_set:
                    rr = 1.0 / rank
                    break
            metrics_by_k[k]["mrr"].append(rr)

            # HR@K
            metrics_by_k[k]["hit_rate"].append(1.0 if hits else 0.0)

    # Aggregate results
    catalog_size = len(recommender.smd)
    results = {"n_users": len(profiles), "catalog_size": catalog_size}

    for k in k_values:
        for metric_name, values in metrics_by_k[k].items():
            arr = np.array(values)
            results[f"{metric_name}@{k}"] = round(float(arr.mean()), 4)
            results[f"{metric_name}@{k}_std"] = round(float(arr.std()), 4)

    results["coverage"] = round(len(all_recommended) / catalog_size, 4) if catalog_size > 0 else 0.0
    results["ild"] = round(float(np.mean(all_ild_scores)), 4) if all_ild_scores else 0.0
    results["novelty"] = round(float(np.mean(all_novelty_scores)), 4) if all_novelty_scores else 0.0
    results["serendipity"] = round(float(np.mean(all_serendipity_scores)), 4) if all_serendipity_scores else 0.0

    return results


# ---------------------------------------------------------------------------
# Calibration evaluation (Steck 2018)
# ---------------------------------------------------------------------------


def _get_genre_distribution(smd: pd.DataFrame, indices: list[int]) -> dict[str, float]:
    """Return normalized genre frequency distribution for a set of movie indices."""
    counts: Counter = Counter()
    for idx in indices:
        genres = smd.iloc[idx]["genres"]
        if isinstance(genres, list):
            counts.update(genres)
    total = sum(counts.values())
    return {g: c / total for g, c in counts.items()} if total > 0 else {}


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with numerical stability."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-10))))


def evaluate_calibration(
    recommender: ContentBasedRecommender,
    k: int = 10,
    n_users: int = 200,
    alpha_smooth: float = 0.01,
    alpha: float = 0.7,
    mmr_lambda: float = 0.5,
) -> dict:
    """Calibration evaluation (Steck, RecSys 2018).

    Measures whether the genre distribution of recommendations matches
    the user's preference distribution. Uses Jensen-Shannon divergence
    (bounded, symmetric) as the primary metric.

    Lower divergence = better calibrated recommendations.
    """
    print("  Preparing user profiles for calibration...")
    profiles = _prepare_user_profiles(recommender, n_users=n_users)
    print(f"  Evaluating calibration for {len(profiles)} users at K={k}...")

    if not profiles:
        print("  ERROR: No qualifying user profiles found.")
        return {}

    all_kl = []
    all_jsd = []
    genre_over_representation: Counter = Counter()
    genre_under_representation: Counter = Counter()

    smd = recommender.smd

    for profile in profiles:
        profile_idx = profile["profile_indices"]

        # User preference distribution (from profile movies)
        user_dist = _get_genre_distribution(smd, profile_idx)
        if not user_dist:
            continue

        # Recommendation distribution
        rec_indices, _ = recommender.recommend_from_profile(
            profile_idx, n=k, alpha=alpha, mmr_lambda=mmr_lambda,
        )
        if not rec_indices:
            continue

        rec_dist = _get_genre_distribution(smd, rec_indices)
        if not rec_dist:
            continue

        # Align distributions to the same genre set
        all_genres = sorted(set(user_dist) | set(rec_dist))
        p = np.array([user_dist.get(g, 0.0) for g in all_genres])
        q = np.array([rec_dist.get(g, 0.0) for g in all_genres])

        # Smooth q to avoid log(0)
        q_smooth = (1 - alpha_smooth) * q + alpha_smooth * p

        # Re-normalize after smoothing
        q_smooth = q_smooth / q_smooth.sum() if q_smooth.sum() > 0 else q_smooth

        # KL divergence: KL(p || q_smooth)
        kl = _kl_divergence(p, q_smooth)
        all_kl.append(kl)

        # Jensen-Shannon divergence (symmetric, bounded [0, 1])
        m = 0.5 * (p + q_smooth)
        jsd = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q_smooth, m)
        all_jsd.append(jsd)

        # Track per-genre miscalibration (rec_proportion - user_proportion)
        for i, g in enumerate(all_genres):
            diff = q[i] - p[i]
            if diff > 0.05:
                genre_over_representation[g] += 1
            elif diff < -0.05:
                genre_under_representation[g] += 1

    if not all_kl:
        return {}

    # Per-genre miscalibration summary (top over/under represented)
    n_evaluated = len(all_kl)
    miscalibration = {}
    for g, count in genre_over_representation.most_common(5):
        miscalibration[g] = round(count / n_evaluated, 4)

    return {
        "calibration_kl": round(float(np.mean(all_kl)), 4),
        "calibration_kl_std": round(float(np.std(all_kl)), 4),
        "calibration_jsd": round(float(np.mean(all_jsd)), 4),
        "calibration_jsd_std": round(float(np.std(all_jsd)), 4),
        "calibration_score": round(1.0 - float(np.mean(all_jsd)), 4),
        "n_users": n_evaluated,
        "genre_over_represented": miscalibration,
    }


# ---------------------------------------------------------------------------
# Combined evaluation (used by web app offline analytics)
# ---------------------------------------------------------------------------


def evaluate_all(
    recommender: ContentBasedRecommender,
    k: int = 5,
) -> dict:
    """Run ratings-based and calibration evaluation, return unified results.

    Returns dict with keys matching the frontend METRIC_META format:
      - precision_at_k, ndcg_at_k, map_at_k, mrr, hit_rate (from ratings-based @k)
      - coverage, ild, novelty, serendipity (system-level from ratings-based)
      - calibration_score, calibration_jsd, calibration_kl (from calibration eval)
    """
    results = {}

    # Primary: ratings-based evaluation at requested K
    ratings = evaluate_with_ratings(recommender, k_values=[k], n_users=200)
    if ratings:
        results["precision_at_k"] = ratings.get(f"precision@{k}", 0.0)
        results["ndcg_at_k"] = ratings.get(f"ndcg@{k}", 0.0)
        results["map_at_k"] = ratings.get(f"map@{k}", 0.0)
        results["mrr"] = ratings.get(f"mrr@{k}", 0.0)
        results["hit_rate"] = ratings.get(f"hit_rate@{k}", 0.0)
        results["coverage"] = ratings.get("coverage", 0.0)
        results["ild"] = ratings.get("ild", 0.0)
        results["novelty"] = ratings.get("novelty", 0.0)
        results["serendipity"] = ratings.get("serendipity", 0.0)
        results["n_users"] = ratings.get("n_users", 0)

    # Calibration evaluation (Steck 2018)
    cal = evaluate_calibration(recommender, k=k)
    if cal:
        results["calibration_score"] = cal.get("calibration_score", 0.0)
        results["calibration_jsd"] = cal.get("calibration_jsd", 0.0)
        results["calibration_kl"] = cal.get("calibration_kl", 0.0)

    return results


# ---------------------------------------------------------------------------
# Grid search (ratings-based)
# ---------------------------------------------------------------------------


def grid_search_weights(
    df: pd.DataFrame,
    weight_ranges: dict[str, list[float]] | None = None,
    k: int = 10,
    n_users: int = 100,
) -> dict:
    """Grid search over field weights using ratings-based evaluation.

    Optimizes: 0.4*NDCG@K + 0.3*Precision@K + 0.15*ILD + 0.15*Novelty.
    """
    if weight_ranges is None:
        weight_ranges = {
            "genres": [1.0, 1.5, 2.0],
            "keywords": [0.5, 1.0, 1.5],
            "director": [1.5, 2.0, 2.5],
            "collection": [0.5, 1.0, 1.5],
        }

    fields = list(weight_ranges.keys())
    value_lists = [weight_ranges[f] for f in fields]
    combos = list(product(*value_lists))

    print(f"Grid search: {len(combos)} combinations over fields {fields}")

    best_score = -1.0
    best_weights = None
    results = []

    for i, combo in enumerate(combos):
        weights = DEFAULT_WEIGHTS.copy()
        for field, val in zip(fields, combo):
            weights[field] = val

        rec = ContentBasedRecommender(df, weights=weights)
        rec.fit()

        eval_results = evaluate_with_ratings(rec, k_values=[k], n_users=n_users)
        if not eval_results:
            continue

        p = eval_results.get(f"precision@{k}", 0)
        n = eval_results.get(f"ndcg@{k}", 0)
        ild = eval_results.get("ild", 0)
        nov = eval_results.get("novelty", 0)

        # Normalize novelty to [0,1] range (typical range 0-10)
        nov_norm = min(nov / 10.0, 1.0)

        combined = 0.4 * n + 0.3 * p + 0.15 * ild + 0.15 * nov_norm

        result_entry = {
            "weights": dict(zip(fields, combo)),
            f"precision@{k}": p,
            f"ndcg@{k}": n,
            "ild": ild,
            "novelty": nov,
            "combined": round(combined, 4),
        }
        results.append(result_entry)

        if combined > best_score:
            best_score = combined
            best_weights = weights.copy()

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(combos)} done, best so far: {best_score:.4f}")

    if best_weights is None:
        raise RuntimeError("Grid search did not produce any weight configuration")

    print(f"Best combined score: {best_score:.4f}")
    print(f"Best weights: { {f: best_weights[f] for f in fields} }")

    return {
        "best_weights": best_weights,
        "best_score": best_score,
        "results": sorted(results, key=lambda x: x["combined"], reverse=True),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    run_grid = "--grid" in sys.argv

    smd = make_dataset()
    smd = build_features(smd)

    rec = ContentBasedRecommender(smd)
    rec.fit()

    # --- Primary: ratings-based evaluation ---
    print("\n=== Ratings-Based Evaluation (Primary) ===")
    ratings_results = evaluate_with_ratings(rec, k_values=[5, 10])

    if ratings_results:
        for metric, value in ratings_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # --- Calibration evaluation (Steck 2018) ---
    print("\n=== Calibration Evaluation (Steck 2018) ===")
    cal_results = evaluate_calibration(rec, k=10)

    if cal_results:
        print(f"  KL Divergence:     {cal_results['calibration_kl']:.4f} +/- {cal_results['calibration_kl_std']:.4f}")
        print(f"  JSD:               {cal_results['calibration_jsd']:.4f} +/- {cal_results['calibration_jsd_std']:.4f}")
        print(f"  Calibration Score: {cal_results['calibration_score']:.4f} (1 - JSD, higher = better)")
        if cal_results.get("genre_over_represented"):
            print(f"  Over-represented genres: {cal_results['genre_over_represented']}")

    # --- Optional: grid search ---
    if run_grid:
        print("\n=== Grid Search (Ratings-Based) ===")
        grid_results = grid_search_weights(smd, n_users=100)
        print(f"\n  Best weights: {grid_results['best_weights']}")
        print(f"  Best score: {grid_results['best_score']:.4f}")
        print("\n  Top 5 configurations:")
        for r in grid_results["results"][:5]:
            print(f"    {r}")
