"""
Comprehensive evaluation metrics for content-based recommender.
Precision@K, NDCG@K, Coverage, Intra-List Diversity, Novelty, and grid search.
"""

import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

from src.data.make_dataset import make_dataset
from src.features.build_features import build_features
from src.models.predict_model import ContentBasedRecommender, DEFAULT_WEIGHTS


def _genre_overlap(genres_a: list, genres_b: list) -> int:
    """Count genre overlap between two genre lists."""
    return len(set(genres_a) & set(genres_b))


def precision_at_k(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 10
) -> float:
    """Genre-overlap-based Precision@K averaged over test movies."""
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


def ndcg_at_k(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 10
) -> float:
    """NDCG@K using genre overlap count as relevance score."""
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


def coverage(
    recommender: ContentBasedRecommender,
    test_movies: pd.DataFrame,
    catalog_size: int,
    k: int = 10,
) -> float:
    """Fraction of catalog appearing in any recommendation list."""
    all_rec_titles = set()

    for _, row in test_movies.iterrows():
        recs = recommender.recommend(row["title"], n=k)
        if not recs.empty:
            all_rec_titles.update(recs["title"].tolist())

    return len(all_rec_titles) / catalog_size if catalog_size > 0 else 0.0


def intra_list_diversity(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 10
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


def novelty(
    recommender: ContentBasedRecommender,
    test_movies: pd.DataFrame,
    popularity_scores: pd.Series,
    k: int = 10,
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


def serendipity_at_k(
    recommender: ContentBasedRecommender,
    test_movies: pd.DataFrame,
    popularity_scores: pd.Series,
    k: int = 10,
) -> float:
    """Serendipity@K = mean(unexpectedness * relevance).
    unexpectedness = 1 - normalized_popularity, relevance = genre_overlap > 0."""
    pop_max = popularity_scores.max()
    if pop_max == 0:
        return 0.0

    pop_norm = popularity_scores / pop_max
    pop_by_title = pd.Series(pop_norm.values, index=recommender.smd["title"])
    pop_by_title = pop_by_title[~pop_by_title.index.duplicated(keep="first")]

    scores = []
    for _, row in test_movies.iterrows():
        title = row["title"]
        true_genres = row["genres"] if isinstance(row["genres"], list) else []
        if not true_genres:
            continue

        recs = recommender.recommend(title, n=k)
        if recs.empty:
            continue

        for _, rec_row in recs.iterrows():
            rec_genres = (
                rec_row["genres"] if isinstance(rec_row["genres"], list) else []
            )
            relevance = 1.0 if _genre_overlap(true_genres, rec_genres) > 0 else 0.0
            p = float(pop_by_title.get(rec_row["title"], 0.5))
            unexpectedness = 1.0 - p
            scores.append(unexpectedness * relevance)

    return float(np.mean(scores)) if scores else 0.0


def mean_reciprocal_rank(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 10
) -> float:
    """MRR = mean(1/rank_of_first_relevant_item). Relevance = genre_overlap > 0."""
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


def per_genre_precision(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 10
) -> dict[str, float]:
    """Precision@K broken down by genre. Only genres with 3+ test movies."""
    genre_results: dict[str, list[float]] = {}

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
        prec = relevant / k

        for g in true_genres:
            genre_results.setdefault(g, []).append(prec)

    return {
        g: round(float(np.mean(v)), 4) for g, v in genre_results.items() if len(v) >= 3
    }


def evaluate_all(
    recommender: ContentBasedRecommender, test_movies: pd.DataFrame, k: int = 10
) -> dict[str, float]:
    """Run all evaluation metrics and return results dict."""
    catalog_size = len(recommender.smd)
    pop_scores = pd.to_numeric(recommender.smd["popularity"], errors="coerce").fillna(
        0.0
    )

    results = {
        "precision_at_k": precision_at_k(recommender, test_movies, k),
        "ndcg_at_k": ndcg_at_k(recommender, test_movies, k),
        "coverage": coverage(recommender, test_movies, catalog_size, k),
        "intra_list_diversity": intra_list_diversity(recommender, test_movies, k),
        "novelty": novelty(recommender, test_movies, pop_scores, k),
        "serendipity_at_k": serendipity_at_k(recommender, test_movies, pop_scores, k),
        "mrr": mean_reciprocal_rank(recommender, test_movies, k),
        "per_genre_precision": per_genre_precision(recommender, test_movies, k),
    }
    return results


def grid_search_weights(
    df: pd.DataFrame,
    test_movies: pd.DataFrame,
    weight_ranges: dict[str, list[float]] | None = None,
    k: int = 10,
) -> dict:
    """
    Grid search over field weights to optimize Precision@K + NDCG@K.

    Args:
        df: Feature-engineered DataFrame (output of build_features)
        test_movies: Subset of movies for evaluation
        weight_ranges: Dict mapping field → list of weight values to try.
            Defaults to varying genres, keywords, director, collection over [0.5, 1.0, 1.5, 2.0].
        k: Number of recommendations

    Returns:
        Dict with 'best_weights', 'best_score', and 'results' list.
    """
    if weight_ranges is None:
        weight_ranges = {
            "genres": [0.5, 1.0, 1.5, 2.0],
            "keywords": [0.5, 1.0, 1.5, 2.0],
            "director": [1.0, 1.5, 2.0, 2.5],
            "collection": [0.5, 1.0, 1.5, 2.0],
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

        p_at_k = precision_at_k(rec, test_movies, k)
        n_at_k = ndcg_at_k(rec, test_movies, k)
        combined = 0.5 * p_at_k + 0.5 * n_at_k

        result_entry = {
            "weights": dict(zip(fields, combo)),
            "precision_at_k": p_at_k,
            "ndcg_at_k": n_at_k,
            "combined": combined,
        }
        results.append(result_entry)

        if combined > best_score:
            best_score = combined
            best_weights = weights.copy()

        if (i + 1) % 50 == 0:
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


if __name__ == "__main__":
    smd = make_dataset()
    smd = build_features(smd)

    rec = ContentBasedRecommender(smd)
    rec.fit()

    test = smd[smd["vote_count"] >= smd["vote_count"].quantile(0.6)].sample(
        n=min(100, len(smd)), random_state=42
    )

    print("\n=== Evaluation Results ===")
    results = evaluate_all(rec, test, k=10)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
