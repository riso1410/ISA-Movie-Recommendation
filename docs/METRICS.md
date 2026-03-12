# Evaluation Metrics for Content-Based Movie Recommender

## 1. Algorithm Overview

This system is a **pure content-based recommender** that outputs **ranked lists** of movies, not predicted ratings. Understanding this distinction is critical for choosing appropriate evaluation metrics.

### How It Works

1. **Feature extraction**: 8 text fields (overview, genres, keywords, cast, director, decade, language, collection) each processed by a separate TF-IDF or Count vectorizer
2. **Weighted combination**: Each field's sparse matrix is multiplied by its weight, then horizontally stacked into one feature matrix
3. **Similarity**: Pairwise cosine similarity computed over the combined feature matrix
4. **Scoring**: `0.7 * cosine_similarity + 0.3 * normalized_weighted_rating` (IMDB formula)
5. **Diversity**: MMR (Maximal Marginal Relevance) re-ranking with lambda=0.5

**Output**: An ordered list of N recommended movies — not numeric rating predictions.

### Why Metric Choice Matters

The system does **not** predict ratings (e.g., "User X will rate Movie Y 4.2 stars"). It produces a ranked list of similar movies given a seed movie. Metrics designed for rating prediction (regression metrics) are fundamentally incompatible. Metrics designed for ranked retrieval are the correct choice.

Additionally, there are **no real user interaction logs** available offline. Ground truth must be approximated using content similarity proxies (genre overlap). This is a well-known limitation of offline content-based evaluation (Castells et al., 2022).

---

## 2. Metric-by-Metric Analysis

### 2.1 Metrics from the Evaluation Checklist

#### Precision@K — APPLICABLE (Implemented)

**Definition**: Fraction of top-K recommendations that are relevant.

$$\text{Precision@K} = \frac{|\text{relevant items in top-K}|}{K}$$

**In our system**: A recommendation is "relevant" if it shares at least 1 genre with the seed movie.

**Why applicable**: Directly measures whether the ranked output contains relevant items. The most fundamental metric for any recommender producing ranked lists.

**Current result**: **0.992** (K=10) — 99.2% of recommendations share at least one genre with the seed.

**Limitations**: Binary relevance (relevant/not) ignores degree of relevance. A movie sharing 4 genres counts the same as one sharing 1 genre. This is addressed by NDCG.

---

#### Accuracy — NOT APPLICABLE

**Definition**: Fraction of all predictions that are correct: `(TP + TN) / (TP + TN + FP + FN)`.

**Why not applicable**: Recommenders face extreme class imbalance. With a catalog of ~9,000 movies and K=10 recommendations:

- ~8,990 movies are correctly "not recommended" (true negatives)
- Accuracy ≈ 8990/9000 ≈ **99.9%** regardless of recommendation quality

A system that recommends 10 random movies would achieve ~99.9% accuracy. This metric is **vacuously high** and universally rejected for recommender evaluation in the literature (Herlocker et al., 2004).

---

#### Recall@K — APPLICABLE (Not Implemented)

**Definition**: Fraction of all relevant items that appear in the top-K recommendations.

$$\text{Recall@K} = \frac{|\text{relevant items in top-K}|}{|\text{total relevant items}|}$$

**Why applicable in principle**: Measures how much of the relevant catalog the system surfaces.

**Practical limitation**: For content-based systems, the number of "relevant" items (genre-matching movies) is typically 500–2000+ per query. With K=10, Recall@10 will always be tiny (10/1000 ≈ 1%). The metric is computable but produces uninformatively small numbers.

**Priority**: MEDIUM — more useful for understanding breadth than for comparing systems.

---

#### F1@K — LOW PRIORITY

**Definition**: Harmonic mean of Precision@K and Recall@K.

$$\text{F1@K} = 2 \cdot \frac{\text{Precision@K} \cdot \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}}$$

**Why low priority**: F1 treats all positions in the ranked list equally — position 1 and position 10 contribute the same. NDCG@K is strictly superior because it accounts for rank position (earlier = more valuable). Given that NDCG is already implemented and provides more information, F1@K adds little value.

---

#### MAE (Mean Absolute Error) — NOT APPLICABLE

**Definition**: Average absolute difference between predicted and actual ratings.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Why not applicable**: MAE requires:

1. **Predicted ratings** — our system outputs ranked lists with similarity scores, not rating predictions
2. **Ground truth ratings per user** — the pipeline has no per-user rating data in the recommendation loop

MAE is designed for collaborative filtering systems that predict "User X will rate Movie Y at 4.2 stars" (e.g., Netflix Prize). Our cosine similarity score (0.0–1.0) is not a rating prediction — it measures content overlap between movies. Comparing `similarity=0.85` to `user_rating=4.5` is meaningless.

---

#### MSE (Mean Squared Error) — NOT APPLICABLE

**Definition**: Average squared difference between predicted and actual ratings.

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Why not applicable**: Same fundamental issue as MAE. MSE (and its root, RMSE) was the primary metric for the Netflix Prize, which was a **rating prediction** problem. Our system solves a different problem: ranking items by content similarity. No rating prediction exists to compute MSE against.

The squared penalty in MSE is designed to penalize large rating errors more heavily (predicting 1 star for a 5-star movie). This penalty structure has no meaning when the output is a ranked list.

---

#### Log Loss (Binary Cross-Entropy) — NOT APPLICABLE

**Definition**: Measures the quality of probabilistic predictions.

$$\text{Log Loss} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i)]$$

**Why not applicable**: Log Loss requires:

1. **Calibrated probability output** P(relevant | item) — our similarity scores are not probabilities
2. **Binary labels** (relevant / not relevant) for all items

Cosine similarity is not a probability. A similarity of 0.7 does not mean "70% chance the user will like this." Converting similarity scores to pseudo-probabilities (e.g., via sigmoid) would be mathematically possible but meaningless — the resulting Log Loss would evaluate the calibration of an arbitrary transformation, not the quality of recommendations.

To use Log Loss properly, one would need to train a separate binary classifier layer on top of the features, with labeled interaction data. This is a different system architecture entirely.

---

#### Computation Time — APPLICABLE (Operational Metric)

**Definition**: Wall-clock time for model fitting and query serving.

**Why applicable**: Not a quality metric, but important for production viability. Should be tracked as two separate measurements:

| Operation                                       | Current Performance |
| ----------------------------------------------- | ------------------- |
| `fit()` — vectorize + compute cosine similarity | ~0.4s               |
| `recommend()` — single query                    | <50ms               |

**Note**: Computation time scales with catalog size. The current O(n^2) cosine similarity precomputation works for ~9,000 movies but would need approximate nearest neighbor methods (e.g., FAISS, Annoy) for larger catalogs.

---

### 2.2 Already Implemented Metrics (Beyond Checklist)

#### NDCG@K (Normalized Discounted Cumulative Gain) — IMPLEMENTED

**Definition**: Measures ranking quality with graded (non-binary) relevance, discounted by position.

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

**In our system**: Relevance = number of overlapping genres between recommendation and seed movie (graded: 0, 1, 2, 3...). IDCG is computed from the ideal ranking (sorted by relevance).

**Why it's the best single metric**: NDCG captures both relevance (how good are the items?) and ranking quality (are better items ranked higher?). It's the standard metric for information retrieval and ranked recommendation evaluation.

**Current result**: **0.965** (K=10)

---

#### Coverage — IMPLEMENTED

**Definition**: Fraction of the total catalog that appears in at least one recommendation list across all test queries.

$$\text{Coverage} = \frac{|\bigcup_q \text{recs}(q)|}{|\text{catalog}|}$$

**Current result**: **0.151** (15.1% of ~9,000 movies)

**Interpretation**: Low coverage indicates popularity bias — the system tends to recommend the same popular movies. This is a known tradeoff in content-based systems where popular movies have richer metadata.

---

#### Intra-List Diversity (ILD) — IMPLEMENTED

**Definition**: Average dissimilarity within recommendation lists.

$$\text{ILD} = 1 - \frac{1}{|S| \choose 2} \sum_{i,j \in S, i < j} \text{cosine\_sim}(i, j)$$

**Current result**: **0.729**

**Interpretation**: Directly validates the MMR re-ranking component. Without MMR, recommendations would be near-duplicates (ILD close to 0). The 0.729 score shows MMR successfully diversifies while maintaining relevance.

---

#### Novelty — IMPLEMENTED

**Definition**: Average self-information (inverse popularity) of recommended items.

$$\text{Novelty} = \frac{1}{|L|} \sum_{i \in L} -\log_2(p(i))$$

where p(i) is the normalized popularity of item i.

**Current result**: **6.099**

**Interpretation**: Higher novelty = less popular recommendations on average. Measures long-tail discovery — whether the system can recommend lesser-known but relevant movies rather than always suggesting blockbusters.

---

#### MRR (Mean Reciprocal Rank) — IMPLEMENTED

**Definition**: Average of the reciprocal rank of the first relevant item.

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

**Interpretation**: Measures how quickly the system surfaces a relevant result. MRR = 1.0 means the first recommendation is always relevant. Given our Precision@10 of 0.992, MRR is expected to be very close to 1.0.

---

#### Serendipity@K — IMPLEMENTED

**Definition**: Measures relevant but unexpected recommendations.

$$\text{Serendipity@K} = \frac{1}{K} \sum_{i=1}^{K} \text{unexpectedness}(i) \times \text{relevance}(i)$$

where unexpectedness = 1 - normalized_popularity, relevance = 1 if genre overlap > 0.

**Interpretation**: Rewards the system for recommending relevant movies that are not obvious popular choices. Combines relevance with novelty at the item level.

---

### 2.3 Additional Metrics (Not Implemented)

#### MAP (Mean Average Precision) — NOT IMPLEMENTED

**Definition**: Mean of Average Precision (AP) across all queries. AP rewards systems that place relevant items earlier in the ranked list.

$$\text{AP@K} = \frac{1}{\min(K, |\text{relevant}|)} \sum_{k=1}^{K} \text{Precision@k} \times \text{rel}(k)$$

$$\text{MAP@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP@K}(q)$$

where rel(k) = 1 if item at rank k is relevant, 0 otherwise.

**How it works**: For each query, AP computes precision at every position where a relevant item appears, then averages those values. This means placing a relevant item at position 1 contributes more than placing it at position 5. MAP then averages AP across all test queries.

**Example**: For a list of 5 items where R = relevant, N = not relevant:

- List: [R, N, R, N, R] → AP = (1/1 + 2/3 + 3/5) / 3 = 0.756
- List: [N, N, R, R, R] → AP = (1/3 + 2/4 + 3/5) / 3 = 0.511
- Same 3 relevant items, but the first ranking scores higher because relevant items appear earlier.

**Why applicable in principle**: MAP is a standard information retrieval metric for ranked lists. It uses binary relevance (relevant/not), which aligns with our genre overlap threshold (overlap > 0 = relevant).

**Why low priority for this system**: MAP and NDCG@K both measure ranking quality, but they differ in relevance handling:

- **MAP** uses **binary** relevance (relevant or not)
- **NDCG** uses **graded** relevance (genre overlap count: 0, 1, 2, 3+)

Since our evaluation already computes graded relevance (genre overlap count), NDCG extracts more information from the same data. Additionally, with Precision@10 = 0.992, nearly all items in every list are relevant, which means AP values will be very close to 1.0 across queries — making MAP uninformative for differentiating system variants.

MAP would become more useful if:

- A stricter relevance threshold were adopted (e.g., overlap >= 2 genres)
- The system were compared against weaker baselines where ranking order matters more

**Priority**: LOW — redundant given NDCG and near-saturated precision

---

#### Hit Rate (Hit@K)

**Definition**: Fraction of queries where at least one recommendation in the top-K is relevant.

**Status**: Not implemented. With Precision@10 = 0.992, Hit Rate will saturate at ~1.0 (almost every query has at least 1 relevant recommendation). Uninformative for this system.

**Priority**: LOW

---

## 3. The Ground Truth Problem

### No User Interaction Data

The system has no access to:

- User click/watch/rating logs
- Implicit feedback (view time, scroll depth)
- A/B test results

All offline evaluation uses **genre overlap** as a proxy for relevance. This is the standard approach for content-based systems without user data, but it has fundamental limitations.

### Genre Overlap as Proxy

**How it works**: A recommendation for "The Dark Knight" (Action, Crime, Drama, Thriller) is considered relevant if it shares at least 1 genre. Genre overlap count (0–4+) provides graded relevance for NDCG.

**Strengths**:

- Genre is the most consistent and complete metadata field
- Captures broad thematic similarity
- Computable for all movies in the catalog

**Limitations**:

- **Overestimates quality**: Genre similarity is necessary but not sufficient. Two Action movies can be completely different in tone, era, and appeal
- **Misses cross-genre appeal**: A user who likes "The Dark Knight" might enjoy "Zodiac" (Crime, Drama, Mystery) — not a genre match for the Thriller component but thematically very similar
- **Circular reasoning risk**: The recommender uses genres as one of its 8 feature fields. Evaluating on genre overlap partially measures "does the system use genre information?" rather than "does the system recommend good movies?"
- **No personalization signal**: Without real users, we cannot evaluate whether recommendations actually satisfy individual preferences

### What True Evaluation Requires

Robust evaluation of this system would require:

1. **Online A/B testing**: Deploy two system variants, measure real user engagement (click-through rate, watch completion, return visits)
2. **User studies**: Small-scale qualitative evaluation where users rate recommendation quality
3. **Held-out interaction data**: If user ratings/watches were available, hold out a test set and measure whether recommendations predict future interactions

The offline metrics implemented here are **necessary for development** (catching regressions, comparing variants, validating components like MMR) but **insufficient for production validation**.

---

## 4. Current Implementation Status

### Implemented in `src/models/evaluate_model.py`

| Metric               | Function                 | Status                                   |
| -------------------- | ------------------------ | ---------------------------------------- |
| Precision@K          | `precision_at_k()`       | Computed, reported                       |
| NDCG@K               | `ndcg_at_k()`            | Computed, reported                       |
| Coverage             | `coverage()`             | Computed, reported                       |
| Intra-List Diversity | `intra_list_diversity()` | Computed, reported                       |
| Novelty              | `novelty()`              | Computed, reported                       |
| MRR                  | `mean_reciprocal_rank()` | Computed, reported                       |
| Serendipity@K        | `serendipity_at_k()`     | Computed, reported                       |
| Per-Genre Precision  | `per_genre_precision()`  | Computed, reported                       |
| Grid Search          | `grid_search_weights()`  | Optimizes field weights via P@K + NDCG@K |

### Current Results (V4, K=10, 200 test movies)

| Metric               | Score |
| -------------------- | ----- |
| Precision@10         | 0.992 |
| NDCG@10              | 0.965 |
| Coverage             | 0.151 |
| Intra-List Diversity | 0.729 |
| Novelty              | 6.099 |

### Not Applicable (Excluded with Justification)

| Metric   | Reason                                                 |
| -------- | ------------------------------------------------------ |
| MAE      | System outputs ranked lists, not rating predictions    |
| MSE      | Same as MAE — requires rating prediction architecture  |
| Log Loss | Similarity scores are not calibrated probabilities     |
| Accuracy | Vacuously high (~99.9%) due to extreme class imbalance |

---

## 5. Recommendations for Improvement

### Short-Term (Offline)

1. **Recall@K**: Implement to understand recommendation breadth, even though absolute values will be small
2. **Coverage improvement**: Current 15.1% indicates significant popularity bias. Consider increasing MMR lambda or adding popularity-penalized scoring
3. **Cross-validation**: Run evaluation over multiple random seeds (not just random_state=42) to estimate metric variance
4. **Non-genre ground truth**: Explore using keyword overlap or director overlap as additional relevance proxies to reduce circular evaluation

### Medium-Term (If User Data Becomes Available)

5. **Implicit feedback metrics**: If the web app logs swipe data, use liked/disliked movies as ground truth instead of genre overlap
6. **Online metrics**: Track click-through rate, session length, and "recommendation accepted" rate from the Tinder-style UI
7. **Temporal evaluation**: Split user interactions by time — train on past, evaluate on future

### Long-Term (Production)

8. **A/B testing framework**: Compare algorithm variants on real user engagement
9. **User satisfaction surveys**: Direct quality measurement
10. **Hybrid evaluation**: Combine offline metrics (for rapid iteration) with online metrics (for validation)

---

## References

- Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). Evaluating collaborative filtering recommender systems. _ACM Transactions on Information Systems_, 22(1), 5–53.
- Castells, P., Hurley, N. J., & Vargas, S. (2022). Novelty and diversity in recommender systems. In _Recommender Systems Handbook_ (pp. 603–646). Springer.
- Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. _SIGIR '98_.
- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. _ACM Transactions on Information Systems_, 20(4), 422–446.
