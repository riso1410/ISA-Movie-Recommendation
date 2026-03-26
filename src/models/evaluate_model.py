"""
Evaluation metrics for content-based recommender.

Two evaluation modes:
  1. Ratings-based (primary): Leave-N-out with real MovieLens user ratings as ground truth.
  2. Genre-based (diagnostic): Genre overlap proxy — kept for coherence diagnostics only.

Metrics: Precision@K, Recall@K, NDCG@K, MAP@K, MRR, HR@K, Coverage, ILD, Novelty, Serendipity.
"""

import sys
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity

from src.data.make_dataset import make_dataset, load_ratings
from src.features.build_features import build_features
from src.models.predict_model import ContentBasedRecommender, DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Ratings-based evaluation (primary)
# ---------------------------------------------------------------------------


def _build_tmdb_to_idx(recommender: ContentBasedRecommender) -> dict[int, int]:
    """Map TMDB movie ID → index in recommender.smd."""
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

    print(f"  Preparing user profiles...")
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
# Genre-based evaluation (diagnostic — kept for coherence checks)
# ---------------------------------------------------------------------------


def _genre_overlap(genres_a: list, genres_b: list) -> int:
    """Count genre overlap between two genre lists."""
    return len(set(genres_a) & set(genres_b))


def genre_precision_at_k(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 5
) -> float:
    """Genre-overlap-based Precision@K averaged over test movies (diagnostic)."""
    precisions = []

    for _, row in test_movies.iterrows():
        title = row["title"]
        true_genres = row["genres"] if isinstance(row["genres"], list) else []
        if not true_genres:
            continue

        recs = recommender.recommend(title, n=k)
        if recs.empty:
            continue

        relevant = 0
        for _, rec_row in recs.iterrows():
            rec_genres = (
                rec_row["genres"] if isinstance(rec_row["genres"], list) else []
            )
            if _genre_overlap(true_genres, rec_genres) > 0:
                relevant += 1

        precisions.append(relevant / k)

    return float(np.mean(precisions)) if precisions else 0.0


def genre_ndcg_at_k(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 5
) -> float:
    """NDCG@K using genre overlap count as relevance score (diagnostic)."""
    from sklearn.metrics import ndcg_score

    ndcg_scores = []

    for _, row in test_movies.iterrows():
        title = row["title"]
        true_genres = row["genres"] if isinstance(row["genres"], list) else []
        if not true_genres:
            continue

        recs = recommender.recommend(title, n=k)
        if recs.empty:
            continue

        relevances = []
        for _, rec_row in recs.iterrows():
            rec_genres = (
                rec_row["genres"] if isinstance(rec_row["genres"], list) else []
            )
            relevances.append(_genre_overlap(true_genres, rec_genres))

        while len(relevances) < k:
            relevances.append(0)

        if max(relevances) == 0:
            continue

        true_relevance = np.array([sorted(relevances, reverse=True)])
        pred_relevance = np.array([relevances])
        ndcg_scores.append(ndcg_score(true_relevance, pred_relevance, k=k))

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def genre_coverage(
    recommender: ContentBasedRecommender,
    test_movies: pd.DataFrame,
    catalog_size: int,
    k: int = 5,
) -> float:
    """Fraction of catalog appearing in any recommendation list."""
    all_rec_titles = set()

    for _, row in test_movies.iterrows():
        recs = recommender.recommend(row["title"], n=k)
        if not recs.empty:
            all_rec_titles.update(recs["title"].tolist())

    return len(all_rec_titles) / catalog_size if catalog_size > 0 else 0.0


def genre_intra_list_diversity(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 5
) -> float:
    """Average (1 - mean pairwise cosine similarity) within recommendation lists."""
    cosine_sim = recommender.cosine_sim
    if cosine_sim is None:
        raise RuntimeError("Recommender must be fitted before evaluation")

    diversities = []

    for _, row in test_movies.iterrows():
        title = row["title"]
        recs = recommender.recommend(title, n=k)
        if len(recs) < 2:
            continue

        rec_titles = recs["title"].tolist()
        rec_indices = []
        for t in rec_titles:
            if t in recommender.indices:
                rec_indices.append(recommender.indices[t])

        if len(rec_indices) < 2:
            continue

        sub_sim = cosine_sim[np.ix_(rec_indices, rec_indices)]
        n_items = len(rec_indices)
        upper_mask = np.triu_indices(n_items, k=1)
        mean_sim = sub_sim[upper_mask].mean()
        diversities.append(1.0 - mean_sim)

    return float(np.mean(diversities)) if diversities else 0.0


def genre_novelty(
    recommender: ContentBasedRecommender,
    test_movies: pd.DataFrame,
    popularity_scores: pd.Series,
    k: int = 5,
) -> float:
    """Average inverse popularity (self-information) of recommended items."""
    pop_max = popularity_scores.max()
    if pop_max == 0:
        return 0.0

    pop_norm = popularity_scores / pop_max
    pop_by_title = pd.Series(pop_norm.values, index=recommender.smd["title"])
    pop_by_title = pop_by_title[~pop_by_title.index.duplicated(keep="first")]

    novelty_scores = []

    for _, row in test_movies.iterrows():
        recs = recommender.recommend(row["title"], n=k)
        if recs.empty:
            continue

        for _, rec_row in recs.iterrows():
            p = float(pop_by_title.get(rec_row["title"], 0.5))
            novelty_scores.append(-np.log2(max(p, 1e-10)))

    return float(np.mean(novelty_scores)) if novelty_scores else 0.0


def genre_map_at_k(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 5
) -> float:
    """MAP@K with genre overlap as relevance (diagnostic). Fixed denominator."""
    ap_scores = []

    for _, row in test_movies.iterrows():
        title = row["title"]
        true_genres = row["genres"] if isinstance(row["genres"], list) else []
        if not true_genres:
            continue

        recs = recommender.recommend(title, n=k)
        if recs.empty:
            continue

        # Count total relevant items in catalog for this query (capped at k)
        all_genres = recommender.smd["genres"]
        total_relevant = sum(
            1 for g in all_genres
            if isinstance(g, list) and _genre_overlap(true_genres, g) > 0
        )
        total_relevant -= 1  # exclude the query movie itself

        hits = 0
        sum_precisions = 0.0
        for rank, (_, rec_row) in enumerate(recs.iterrows(), 1):
            rec_genres = (
                rec_row["genres"] if isinstance(rec_row["genres"], list) else []
            )
            if _genre_overlap(true_genres, rec_genres) > 0:
                hits += 1
                sum_precisions += hits / rank

        denom = min(k, total_relevant) if total_relevant > 0 else 1
        ap_scores.append(sum_precisions / denom)

    return float(np.mean(ap_scores)) if ap_scores else 0.0


def genre_mrr(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 5
) -> float:
    """MRR with genre overlap as relevance (diagnostic)."""
    rr_scores = []

    for _, row in test_movies.iterrows():
        title = row["title"]
        true_genres = row["genres"] if isinstance(row["genres"], list) else []
        if not true_genres:
            continue

        recs = recommender.recommend(title, n=k)
        if recs.empty:
            continue

        for rank, (_, rec_row) in enumerate(recs.iterrows(), 1):
            rec_genres = (
                rec_row["genres"] if isinstance(rec_row["genres"], list) else []
            )
            if _genre_overlap(true_genres, rec_genres) > 0:
                rr_scores.append(1.0 / rank)
                break
        else:
            rr_scores.append(0.0)

    return float(np.mean(rr_scores)) if rr_scores else 0.0


def genre_evaluate_all(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 5
) -> dict[str, float]:
    """Run all genre-based diagnostic metrics."""
    catalog_size = len(recommender.smd)
    pop_scores = pd.to_numeric(
        recommender.smd["popularity"], errors="coerce"
    ).fillna(0.0)

    return {
        "genre_precision_at_k": genre_precision_at_k(recommender, test_movies, k),
        "genre_ndcg_at_k": genre_ndcg_at_k(recommender, test_movies, k),
        "genre_map_at_k": genre_map_at_k(recommender, test_movies, k),
        "genre_mrr": genre_mrr(recommender, test_movies, k),
        "genre_coverage": genre_coverage(recommender, test_movies, catalog_size, k),
        "genre_ild": genre_intra_list_diversity(recommender, test_movies, k),
        "genre_novelty": genre_novelty(recommender, test_movies, pop_scores, k),
    }


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

    # --- Diagnostic: genre-based evaluation ---
    print("\n=== Genre-Based Evaluation (Diagnostic) ===")
    test = smd.sample(n=min(100, len(smd)), random_state=42)

    for k in [5, 10]:
        print(f"\n  K={k}:")
        genre_results = genre_evaluate_all(rec, test, k=k)
        for metric, value in genre_results.items():
            if isinstance(value, float):
                print(f"    {metric}: {value:.4f}")

    # --- Optional: grid search ---
    if run_grid:
        print("\n=== Grid Search (Ratings-Based) ===")
        grid_results = grid_search_weights(smd, n_users=100)
        print(f"\n  Best weights: {grid_results['best_weights']}")
        print(f"  Best score: {grid_results['best_score']:.4f}")
        print(f"\n  Top 5 configurations:")
        for r in grid_results["results"][:5]:
            print(f"    {r}")
