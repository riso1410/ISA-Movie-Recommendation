"""
Content-based recommendation engine.
TF-IDF + Cosine Similarity + Weighted Rating Filter.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    """Content-based movie recommender using TF-IDF and cosine similarity."""

    def __init__(self, smd):
        self.smd = smd.reset_index(drop=True)
        self.indices = pd.Series(self.smd.index, index=self.smd['title']).drop_duplicates()
        self.cosine_sim = None
        self.C = self.smd['vote_average'].mean()
        self.m = self.smd['vote_count'].quantile(0.60)
        self._compute_weighted_ratings()

    def _compute_weighted_ratings(self):
        """Calculate IMDB-style weighted ratings."""
        m, C = self.m, self.C
        self.smd['weighted_rating'] = self.smd.apply(
            lambda x: (x['vote_count'] / (x['vote_count'] + m)) * x['vote_average']
                      + (m / (x['vote_count'] + m)) * C,
            axis=1
        )

    def fit(self, max_features=20000):
        """Fit TF-IDF and compute cosine similarity matrix."""
        tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
        tfidf_matrix = tfidf.fit_transform(self.smd['soup'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        return self

    def recommend(self, title, n=10):
        """
        Get top-n recommendations for a given movie title.
        Uses cosine similarity + weighted rating filter.
        """
        if title not in self.indices:
            return pd.DataFrame()

        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n * 3 + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        candidates = self.smd[['title', 'genres', 'vote_average', 'vote_count', 'weighted_rating']].iloc[movie_indices].copy()
        candidates['similarity_score'] = scores

        qualified = candidates[candidates['vote_count'] >= self.m]
        qualified = qualified.sort_values('weighted_rating', ascending=False)

        return qualified.head(n).reset_index(drop=True)


def build_recommender(smd):
    """Convenience function to build and fit the recommender."""
    recommender = ContentBasedRecommender(smd)
    recommender.fit()
    return recommender
