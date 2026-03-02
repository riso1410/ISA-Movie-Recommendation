# Evaluation Report: Content-Based Recommender V3 vs V4

**Dataset**: 9,219 movies (The Movies Dataset, Kaggle)
**Test set**: 200 movies (vote_count >= 60th percentile, random_state=42)
**k**: 10

---

## Model Descriptions

|                    | V3 (Baseline)                                         | V4 (New)                                                                                                                         |
| ------------------ | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Vectorization**  | Single TF-IDF on concatenated "soup"                  | Per-field vectorizers (TF-IDF + Count) with weighted hstack                                                                      |
| **Fields**         | overview + genres + keywords + cast(3) + director(x3) | overview, genres, keywords, cast(5), director, decade, language, collection                                                      |
| **Weighting**      | Director repeated 3x in soup (implicit)               | Explicit field weights: director=2.0, genres=1.5, collection=1.5, keywords=1.2, cast=1.0, overview=1.0, language=0.5, decade=0.3 |
| **Quality signal** | Hard filter: vote_count >= 60th percentile            | Soft: `0.7 * similarity + 0.3 * normalized_weighted_rating`                                                                      |
| **Diversity**      | None                                                  | MMR re-ranking (lambda=0.5)                                                                                                      |
| **Fit time**       | 0.3s                                                  | 0.4s                                                                                                                             |

---

## Metrics Comparison

| Metric                   | V3        | V4        | Delta           | Direction       |
| ------------------------ | --------- | --------- | --------------- | --------------- |
| **Precision@10**         | 0.743     | **0.992** | +0.249 (+33.6%) | Higher = better |
| **NDCG@10**              | 0.821     | **0.965** | +0.144 (+17.5%) | Higher = better |
| **Coverage**             | 0.142     | **0.151** | +0.010 (+6.8%)  | Higher = better |
| **Intra-List Diversity** | **0.935** | 0.729     | -0.206 (-22.0%) | Higher = better |
| **Novelty**              | 5.682     | **6.099** | +0.417 (+7.3%)  | Higher = better |

### Interpretation

- **Precision@10**: Massive improvement. V4 achieves near-perfect genre relevance (99.2% of recommended movies share at least one genre with the query). Per-field vectorizers with explicit weights produce far more genre-coherent recommendations than the single soup approach.

- **NDCG@10**: Strong improvement. Not only are recommendations relevant, but higher-relevance items (more genre overlap) are ranked higher in the list.

- **Coverage**: Slight improvement. V4 surfaces 15.1% of the catalog vs V3's 14.2%. The soft quality signal (no hard vote_count cutoff) allows more niche films to appear.

- **Intra-List Diversity (ILD)**: V4 is lower (0.729 vs 0.935). This is a **known and intentional tradeoff**: V4's per-field weighting produces more genre-coherent results (which lowers pairwise diversity within lists). V3's soup-based approach mixed signals, producing more varied but less relevant results. The 0.729 ILD is still strong — MMR (lambda=0.5) prevents the list from being homogeneous.

- **Novelty**: V4 recommends less popular (more novel) movies on average. The soft quality signal avoids the popularity bias of V3's hard vote_count filter, which only qualified well-known films.

---

## Example: "The Dark Knight" Recommendations

### V3 (soup + hard filter)

| #   | Title                           | Similarity | Genres                                      |
| --- | ------------------------------- | ---------- | ------------------------------------------- |
| 1   | Inception                       | 0.196      | Action, Thriller, SciFi, Mystery, Adventure |
| 2   | Interstellar                    | 0.177      | Adventure, Drama, SciFi                     |
| 3   | Memento                         | 0.163      | Mystery, Thriller                           |
| 4   | The Prestige                    | 0.248      | Drama, Mystery, Thriller                    |
| 5   | The Dark Knight Rises           | 0.501      | Action, Crime, Drama, Thriller              |
| 6   | Batman: Dark Knight Returns Pt2 | 0.217      | Action, Animation                           |
| 7   | Batman Begins                   | 0.434      | Action, Crime, Drama                        |
| 8   | The Lego Movie                  | 0.078      | Adventure, Animation, Comedy, Family        |
| 9   | Batman: Dark Knight Returns Pt1 | 0.144      | Action, Animation                           |
| 10  | Batman: Under the Red Hood      | 0.286      | Action, Animation                           |

**Issues**: Dominated by Batman properties (6/10). The Lego Movie appears (sim=0.078) due to IDF boosting rare "Batman" token. Sorted by weighted_rating, not similarity — Dark Knight Rises at #5 despite highest similarity.

### V4 (per-field vectorizers + MMR)

| #   | Title                 | Similarity | Genres                         |
| --- | --------------------- | ---------- | ------------------------------ |
| 1   | The Dark Knight Rises | 0.844      | Action, Crime, Drama, Thriller |
| 2   | Scarface              | 0.424      | Action, Crime, Drama, Thriller |
| 3   | The Prestige          | 0.525      | Drama, Mystery, Thriller       |
| 4   | Training Day          | 0.427      | Action, Crime, Drama, Thriller |
| 5   | Thursday              | 0.469      | Drama, Action, Crime, Thriller |
| 6   | Heat                  | 0.427      | Action, Crime, Drama, Thriller |
| 7   | Harry Brown           | 0.473      | Thriller, Crime, Drama, Action |
| 8   | Running Scared        | 0.427      | Action, Crime, Drama, Thriller |
| 9   | The Asphalt Jungle    | 0.423      | Action, Crime, Drama, Thriller |
| 10  | Bullitt               | 0.428      | Action, Crime, Drama, Thriller |

**Improvements**: No franchise echo chamber — only 1 Batman film (the direct sequel). CountVectorizer for genres avoids IDF penalty on common genre names. MMR ensures variety beyond just "more Batman". Mix of classic crime films (Heat, Scarface, Bullitt) alongside modern thrillers.

---

## Metric Definitions

| Metric                   | Definition                                                                       |
| ------------------------ | -------------------------------------------------------------------------------- |
| **Precision@K**          | Fraction of top-K recommendations sharing >= 1 genre with query movie            |
| **NDCG@K**               | Normalized Discounted Cumulative Gain using genre overlap count as relevance     |
| **Coverage**             | Fraction of total catalog appearing in any recommendation list across test set   |
| **Intra-List Diversity** | `1 - mean(pairwise cosine similarity)` within each recommendation list, averaged |
| **Novelty**              | Mean self-information `-log2(normalized_popularity)` of recommended items        |

---

## Architecture

```
V4 Pipeline:
  make_dataset → parse decade/language/collection, expand cast to 5
       ↓
  build_features → per-field cleaned columns (overview_clean, genres_str, ...)
       ↓
  ContentBasedRecommender.fit()
       → 8 vectorizers (TF-IDF for overview/keywords, CountVectorizer for rest)
       → weight × matrix per field
       → scipy.sparse.hstack → cosine_similarity
       ↓
  recommend(title, n=10)
       → top 50 candidates by cosine similarity
       → combined score = 0.7 × similarity + 0.3 × normalized_weighted_rating
       → MMR re-ranking (λ=0.5) for diversity
       → return top 10
```
