"""
Content-based recommendation engine.
Supports 3 modes:
  - "overview": TF-IDF on movie overviews only (baseline)
  - "soup":     TF-IDF on concatenated metadata soup
  - "multi":    Per-field TF-IDF/Count vectorizers with weighted combination
All modes use cosine similarity + soft scoring + MMR diversity.
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

MODES = ("overview", "soup", "multi")

MODE_LABELS = {
    "overview": "Overview TF-IDF",
    "soup": "Metadata Soup TF-IDF",
    "multi": "Multi-Vectorizer (Weighted)",
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "overview": 1.0,
    "genres": 1.5,
    "keywords": 1.2,
    "cast": 1.0,
    "director": 2.0,
    "decade": 0.3,
    "language": 0.5,
    "collection": 1.5,
}

FIELD_CONFIG: dict[str, tuple[str, type, dict]] = {
    "overview": (
        "overview_clean",
        TfidfVectorizer,
        {"sublinear_tf": True, "max_features": 15000, "stop_words": "english"},
    ),
    "genres": ("genres_str", CountVectorizer, {}),
    "keywords": (
        "keywords_str",
        TfidfVectorizer,
        {"sublinear_tf": True, "max_features": 5000},
    ),
    "cast": ("cast_str", CountVectorizer, {}),
    "director": ("director_str", CountVectorizer, {}),
    "decade": ("decade", CountVectorizer, {}),
    "language": ("language", CountVectorizer, {}),
    "collection": ("collection", CountVectorizer, {}),
}


class ContentBasedRecommender:
    """Content-based movie recommender with per-field vectorizers and MMR diversity."""

    def __init__(
        self,
        smd: pd.DataFrame,
        weights: dict[str, float] | None = None,
        mode: str = "multi",
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
        self.mode = mode
        self.smd = smd.reset_index(drop=True)
        indices = pd.Series(self.smd.index, index=self.smd["title"])
        self.indices = indices[~indices.index.duplicated(keep="first")]
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.vectorizers: dict[str, TfidfVectorizer | CountVectorizer] = {}
        self.feature_matrix = None
        self.cosine_sim = None

        self.C = self.smd["vote_average"].mean()
        self.m = self.smd["vote_count"].quantile(0.60)
        self._compute_weighted_ratings()

    def _compute_weighted_ratings(self) -> None:
        """Calculate IMDB-style weighted ratings."""
        m, C = self.m, self.C
        v = self.smd["vote_count"]
        R = self.smd["vote_average"]
        self.smd["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C

        wr = self.smd["weighted_rating"]
        wr_min, wr_max = wr.min(), wr.max()
        if wr_max > wr_min:
            self.smd["wr_norm"] = (wr - wr_min) / (wr_max - wr_min)
        else:
            self.smd["wr_norm"] = 0.5

    def fit(self) -> "ContentBasedRecommender":
        """Fit vectorizers and compute cosine similarity based on mode."""
        if self.mode == "overview":
            self._fit_overview()
        elif self.mode == "soup":
            self._fit_soup()
        else:
            self._fit_multi()
        return self

    def _fit_overview(self) -> None:
        """Mode: overview-only TF-IDF."""
        texts = self.smd["overview"].fillna("").astype(str).str.lower()
        vec = TfidfVectorizer(stop_words="english", max_features=20000)
        self.feature_matrix = vec.fit_transform(texts)
        self.vectorizers["overview"] = vec
        self.cosine_sim = cosine_similarity(self.feature_matrix)

    def _fit_soup(self) -> None:
        """Mode: single TF-IDF on concatenated metadata soup."""
        if "soup" not in self.smd.columns:
            raise ValueError("'soup' column missing. Run build_features() first.")
        texts = self.smd["soup"].fillna("").astype(str)
        vec = TfidfVectorizer(stop_words="english", max_features=20000)
        self.feature_matrix = vec.fit_transform(texts)
        self.vectorizers["soup"] = vec
        self.cosine_sim = cosine_similarity(self.feature_matrix)

    def _fit_multi(self) -> None:
        """Mode: per-field weighted multi-vectorizer."""
        matrices = []

        for field, (col, vec_cls, vec_kwargs) in FIELD_CONFIG.items():
            weight = self.weights.get(field, 0.0)
            if weight == 0.0:
                continue

            texts = self.smd[col].fillna("").astype(str)
            vectorizer = vec_cls(**vec_kwargs)
            try:
                matrix = vectorizer.fit_transform(texts)
            except ValueError:
                continue

            matrix = normalize(matrix, norm="l2", axis=1)
            if weight != 1.0:
                matrix = matrix * weight

            self.vectorizers[field] = vectorizer
            matrices.append(matrix)

        self.feature_matrix = hstack(matrices, format="csr")
        self.cosine_sim = cosine_similarity(self.feature_matrix)

    def recommend(
        self, title: str, n: int = 10, alpha: float = 0.7, mmr_lambda: float = 0.5
    ) -> pd.DataFrame:
        """
        Get top-n recommendations with soft scoring and MMR diversity.

        Score = alpha * similarity + (1 - alpha) * normalized_weighted_rating
        MMR re-ranks for diversity: lambda * score - (1-lambda) * max_sim_to_selected
        """
        if title not in self.indices:
            return pd.DataFrame()

        cosine_sim_matrix = self.cosine_sim
        if cosine_sim_matrix is None:
            raise RuntimeError("Recommender must be fitted before calling recommend()")

        idx = self.indices[title]
        n_candidates = n * 5

        sim_scores = cosine_sim_matrix[idx]
        candidate_indices = np.argsort(sim_scores)[::-1][1 : n_candidates + 1]
        candidate_sims = sim_scores[candidate_indices]

        wr_norms = self.smd["wr_norm"].values[candidate_indices]
        combined_scores = alpha * candidate_sims + (1 - alpha) * wr_norms

        selected = self._mmr_rerank(candidate_indices, combined_scores, n, mmr_lambda)

        result_indices = [s[0] for s in selected]
        result_scores = [s[1] for s in selected]
        result_sims = [sim_scores[i] for i in result_indices]

        result = (
            self.smd[
                ["title", "genres", "vote_average", "vote_count", "weighted_rating"]
            ]
            .iloc[result_indices]
            .copy()
        )
        result["similarity_score"] = result_sims
        result["combined_score"] = result_scores

        return result.reset_index(drop=True)

    def _mmr_rerank(
        self, candidate_indices: np.ndarray, scores: np.ndarray, n: int, lam: float
    ) -> list[tuple[int, float]]:
        """Maximal Marginal Relevance re-ranking for diversity."""
        if len(candidate_indices) == 0:
            return []

        cosine_sim_matrix = self.cosine_sim
        if cosine_sim_matrix is None:
            raise RuntimeError("Recommender must be fitted before re-ranking")

        candidate_sim = cosine_sim_matrix[np.ix_(candidate_indices, candidate_indices)]

        remaining = list(range(len(candidate_indices)))
        selected: list[int] = []
        selected_scores: list[float] = []

        for _ in range(min(n, len(candidate_indices))):
            if not remaining:
                break

            if not selected:
                best_local = max(remaining, key=lambda i: scores[i])
            else:
                best_val = -float("inf")
                best_local = remaining[0]
                for i in remaining:
                    max_sim_to_selected = max(candidate_sim[i][j] for j in selected)
                    mmr_score = lam * scores[i] - (1 - lam) * max_sim_to_selected
                    if mmr_score > best_val:
                        best_val = mmr_score
                        best_local = i
            selected.append(best_local)
            selected_scores.append(scores[best_local])
            remaining.remove(best_local)

        return [(candidate_indices[i], s) for i, s in zip(selected, selected_scores)]

    def recommend_from_profile(
        self,
        profile_indices: list[int],
        n: int = 10,
        alpha: float = 0.7,
        mmr_lambda: float = 0.5,
        exclude_indices: set[int] | None = None,
    ) -> tuple[list[int], list[float]]:
        """Recommend from a set of liked movie indices (for evaluation).

        Builds mean feature vector from profile_indices, scores all movies
        via cosine similarity + alpha blend with weighted rating, applies MMR.

        Returns (recommended_indices, combined_scores).
        """
        if self.feature_matrix is None:
            raise RuntimeError("Recommender must be fitted before calling recommend_from_profile()")

        profile = self.feature_matrix[profile_indices].mean(axis=0)
        if hasattr(profile, "A"):
            profile = profile.A

        scores = cosine_similarity(profile.reshape(1, -1), self.feature_matrix).flatten()

        for idx in profile_indices:
            scores[idx] = -np.inf
        if exclude_indices:
            for idx in exclude_indices:
                scores[idx] = -np.inf

        wr_norms = self.smd["wr_norm"].values
        combined = alpha * scores + (1 - alpha) * wr_norms

        n_candidates = n * 5
        valid_mask = combined > -np.inf
        candidate_pool = np.where(valid_mask)[0]
        top_candidates = candidate_pool[np.argsort(combined[candidate_pool])[::-1][:n_candidates]]

        candidate_scores = combined[top_candidates]
        selected = self._mmr_rerank(top_candidates, candidate_scores, n, mmr_lambda)
        return [s[0] for s in selected], [s[1] for s in selected]

    def get_model_stats(self) -> dict:
        """Return model training metadata for analytics."""
        feature_matrix = self.feature_matrix
        stats = {
            "mode": self.mode,
            "mode_label": MODE_LABELS.get(self.mode, self.mode),
            "total_movies": len(self.smd),
            "field_weights": self.weights if self.mode == "multi" else {},
            "weighted_rating_params": {
                "C_mean_vote": round(self.C, 3),
                "m_60th_percentile": round(self.m, 1),
            },
            "vectorizer_stats": {},
            "feature_matrix_shape": list(feature_matrix.shape)
            if feature_matrix is not None
            else None,
            "feature_matrix_sparsity": round(
                1
                - feature_matrix.nnz
                / (feature_matrix.shape[0] * feature_matrix.shape[1]),
                4,
            )
            if feature_matrix is not None
            else None,
            "cosine_sim_stats": {},
            "data_quality": {},
        }

        for field, vec in self.vectorizers.items():
            vocab = getattr(vec, "vocabulary_", {})
            stats["vectorizer_stats"][field] = {
                "type": type(vec).__name__,
                "vocab_size": len(vocab),
                "weight": self.weights.get(field, 0),
            }

        if self.cosine_sim is not None:
            upper = self.cosine_sim[np.triu_indices(len(self.cosine_sim), k=1)]
            sample = np.random.choice(
                upper, size=min(100000, len(upper)), replace=False
            )
            stats["cosine_sim_stats"] = {
                "mean": round(float(np.mean(sample)), 4),
                "median": round(float(np.median(sample)), 4),
                "std": round(float(np.std(sample)), 4),
                "p95": round(float(np.percentile(sample, 95)), 4),
            }

        if self.mode == "multi":
            for field, (col, _, _) in FIELD_CONFIG.items():
                if col in self.smd.columns:
                    empty = (self.smd[col].fillna("").astype(str) == "").sum()
                    stats["data_quality"][field] = {
                        "empty_count": int(empty),
                        "fill_rate": round(1 - empty / len(self.smd), 3),
                    }

        return stats


def build_recommender(
    smd: pd.DataFrame, weights: dict[str, float] | None = None, mode: str = "multi"
) -> ContentBasedRecommender:
    """Convenience function to build and fit the recommender."""
    recommender = ContentBasedRecommender(smd, weights=weights, mode=mode)
    recommender.fit()
    return recommender
