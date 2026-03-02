"""
Content-based recommendation engine.
Per-field TF-IDF/Count vectorizers + cosine similarity + soft scoring + MMR diversity.
"""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_WEIGHTS: dict[str, float] = {
    'overview': 1.0,
    'genres': 1.5,
    'keywords': 1.2,
    'cast': 1.0,
    'director': 2.0,
    'decade': 0.3,
    'language': 0.5,
    'collection': 1.5,
}

# Maps field name → (column in DataFrame, vectorizer class, vectorizer kwargs)
FIELD_CONFIG: dict[str, tuple[str, type, dict]] = {
    'overview':   ('overview_clean', TfidfVectorizer, {'sublinear_tf': True, 'max_features': 15000, 'stop_words': 'english'}),
    'genres':     ('genres_str',     CountVectorizer, {}),
    'keywords':   ('keywords_str',   TfidfVectorizer, {'sublinear_tf': True, 'max_features': 5000}),
    'cast':       ('cast_str',       CountVectorizer, {}),
    'director':   ('director_str',   CountVectorizer, {}),
    'decade':     ('decade',         CountVectorizer, {}),
    'language':   ('language',       CountVectorizer, {}),
    'collection': ('collection',     CountVectorizer, {}),
}


class ContentBasedRecommender:
    """Content-based movie recommender with per-field vectorizers and MMR diversity."""

    def __init__(self, smd: pd.DataFrame, weights: dict[str, float] | None = None):
        self.smd = smd.reset_index(drop=True)
        indices = pd.Series(self.smd.index, index=self.smd['title'])
        self.indices = indices[~indices.index.duplicated(keep='first')]
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.vectorizers: dict[str, object] = {}
        self.feature_matrix = None
        self.cosine_sim = None

        self.C = self.smd['vote_average'].mean()
        self.m = self.smd['vote_count'].quantile(0.60)
        self._compute_weighted_ratings()

    def _compute_weighted_ratings(self) -> None:
        """Calculate IMDB-style weighted ratings."""
        m, C = self.m, self.C
        v = self.smd['vote_count']
        R = self.smd['vote_average']
        self.smd['weighted_rating'] = (v / (v + m)) * R + (m / (v + m)) * C

        # Pre-compute normalized weighted rating for scoring
        wr = self.smd['weighted_rating']
        wr_min, wr_max = wr.min(), wr.max()
        if wr_max > wr_min:
            self.smd['wr_norm'] = (wr - wr_min) / (wr_max - wr_min)
        else:
            self.smd['wr_norm'] = 0.5

    def fit(self) -> 'ContentBasedRecommender':
        """Fit per-field vectorizers and compute combined cosine similarity."""
        matrices = []

        for field, (col, vec_cls, vec_kwargs) in FIELD_CONFIG.items():
            weight = self.weights.get(field, 0.0)
            if weight == 0.0:
                continue

            texts = self.smd[col].fillna('').astype(str)
            vectorizer = vec_cls(**vec_kwargs)
            try:
                matrix = vectorizer.fit_transform(texts)
            except ValueError:
                # Skip fields with empty vocabulary (all docs empty)
                continue

            if weight != 1.0:
                matrix = matrix * weight

            self.vectorizers[field] = vectorizer
            matrices.append(matrix)

        self.feature_matrix = hstack(matrices, format='csr')
        self.cosine_sim = cosine_similarity(self.feature_matrix)
        return self

    def recommend(self, title: str, n: int = 10, alpha: float = 0.7,
                  mmr_lambda: float = 0.5) -> pd.DataFrame:
        """
        Get top-n recommendations with soft scoring and MMR diversity.

        Score = alpha * similarity + (1 - alpha) * normalized_weighted_rating
        MMR re-ranks for diversity: lambda * score - (1-lambda) * max_sim_to_selected
        """
        if title not in self.indices:
            return pd.DataFrame()

        idx = self.indices[title]
        n_candidates = n * 5

        # Get top candidates by cosine similarity
        sim_scores = self.cosine_sim[idx]
        candidate_indices = np.argsort(sim_scores)[::-1][1:n_candidates + 1]
        candidate_sims = sim_scores[candidate_indices]

        # Combined score: similarity + normalized weighted rating
        wr_norms = self.smd['wr_norm'].values[candidate_indices]
        combined_scores = alpha * candidate_sims + (1 - alpha) * wr_norms

        # MMR re-ranking for diversity
        selected = self._mmr_rerank(
            candidate_indices, combined_scores, n, mmr_lambda
        )

        # Build result DataFrame
        result_indices = [s[0] for s in selected]
        result_scores = [s[1] for s in selected]
        result_sims = [sim_scores[i] for i in result_indices]

        result = self.smd[['title', 'genres', 'vote_average', 'vote_count', 'weighted_rating']].iloc[result_indices].copy()
        result['similarity_score'] = result_sims
        result['combined_score'] = result_scores

        return result.reset_index(drop=True)

    def _mmr_rerank(self, candidate_indices: np.ndarray, scores: np.ndarray,
                    n: int, lam: float) -> list[tuple[int, float]]:
        """Maximal Marginal Relevance re-ranking for diversity."""
        if len(candidate_indices) == 0:
            return []

        # Pairwise similarity submatrix among candidates
        candidate_sim = self.cosine_sim[np.ix_(candidate_indices, candidate_indices)]

        remaining = list(range(len(candidate_indices)))
        selected: list[int] = []
        selected_scores: list[float] = []

        for _ in range(min(n, len(candidate_indices))):
            if not remaining:
                break

            if not selected:
                # First pick: highest combined score
                best_local = max(remaining, key=lambda i: scores[i])
            else:
                best_val = -float('inf')
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


def build_recommender(smd: pd.DataFrame, weights: dict[str, float] | None = None) -> ContentBasedRecommender:
    """Convenience function to build and fit the recommender."""
    recommender = ContentBasedRecommender(smd, weights=weights)
    recommender.fit()
    return recommender
