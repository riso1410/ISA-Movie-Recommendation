# How the Recommendation System Works

## Overview

This is a **content-based movie recommendation system**. It recommends movies based on what a movie _is_ (its genres, director, cast, plot, etc.), not based on what other users watched (that would be collaborative filtering). The system has two modes:

1. **Offline**: given a movie title, find similar movies
2. **Online (web app)**: Tinder-style swipe UI that builds a user taste profile in real-time

---

## Step 1: Data Preparation

**Input**: 4 CSV files from Kaggle's "The Movies Dataset" (~9,000 movies after filtering).

The pipeline (`make_dataset.py`) does:

- Loads `movies_metadata.csv`, `credits.csv`, `keywords.csv`, `links_small.csv`
- Cleans bad IDs, filters to the "small" movie subset
- Parses JSON columns: extracts genre names, keyword names, top 5 cast members, director name
- Derives extra fields: **decade** (e.g. `decade_1990s`), **language** (e.g. `lang_en`), **collection** (e.g. franchise name)
- Outputs `movies_processed.csv`

---

## Step 2: Feature Engineering

**Goal**: Turn each movie into text fields that a vectorizer can process.

`build_features.py` cleans each field separately:

- **overview_clean**: lowercased plot description
- **genres_str**: `"action sciencefiction"` (spaces removed so multi-word genres become single tokens)
- **keywords_str**: same treatment as genres
- **cast_str**: `"johnnydepp orlandobloom"` (spaces removed so "Johnny Depp" = one token, not two)
- **director_str**: `"christophernolan"` (same reason)
- **decade, language, collection**: simple string tokens

**Why remove spaces from names?** TF-IDF splits on whitespace. Without this, "Johnny" and "Depp" would be separate tokens — the system couldn't distinguish Johnny Depp from Johnny Knoxville + Jan Depp.

---

## Step 3: Multi-Vectorizer Feature Matrix

This is the core idea. Instead of concatenating all text into one big "soup" string, each field gets **its own vectorizer** with **its own weight**.

```
Field       | Vectorizer     | Weight | Why
------------|----------------|--------|------------------------------------
overview    | TF-IDF (15k)   | 1.0    | Rich text, needs term frequency weighting
genres      | Count          | 1.5    | Small vocabulary, exact match matters
keywords    | TF-IDF (5k)    | 1.2    | Medium vocab, some keywords are more rare/important
cast        | Count          | 1.0    | Either an actor is in the movie or not
director    | Count          | 2.0    | Highest weight - same director = strong signal
decade      | Count          | 0.3    | Weak signal, just a slight era preference
language    | Count          | 0.5    | Minor preference signal
collection  | Count          | 1.5    | Same franchise = very relevant
```

**How it works:**

1. Each field is vectorized independently (fit_transform)
2. Each resulting sparse matrix is multiplied by its weight
3. All matrices are horizontally stacked into one wide sparse matrix: `[overview | genres | keywords | cast | director | decade | language | collection]`
4. Cosine similarity is computed between all movie pairs using this combined matrix

**Why TF-IDF for some, Count for others?** TF-IDF downweights common terms and upweights rare ones — useful for text (overview, keywords) where word frequency matters. For fields like genres or cast, a movie either has it or doesn't, so raw counts (0 or 1) are fine.

**Why per-field vectorizers instead of one big string?** Control. If director matters more than decade, you can't express that in a single string without duplicating tokens. Separate vectorizers + weights give precise control over each signal's influence.

---

## Step 4: IMDB Weighted Rating

Not all movies have reliable ratings. A movie with 1 vote and a 10.0 average isn't truly better than one with 10,000 votes and 8.5.

The system uses the IMDB weighted rating formula:

```
WR = (v / (v + m)) * R + (m / (v + m)) * C
```

- **v** = movie's vote count
- **R** = movie's average vote
- **m** = minimum votes needed (60th percentile of all vote counts)
- **C** = mean vote average across all movies

**Effect**: Movies with few votes get pulled toward the global mean. Movies with many votes keep their actual rating. This is Bayesian smoothing.

The weighted rating is then normalized to [0, 1] range (`wr_norm`).

---

## Step 5: Scoring — Combining Similarity and Quality

When recommending movies, the final score blends content similarity with quality:

```
score = alpha * cosine_similarity + (1 - alpha) * wr_norm
```

Default `alpha = 0.7`, so 70% similarity, 30% quality. This prevents the system from recommending obscure, poorly-rated movies just because they have similar keywords.

---

## Step 6: MMR Diversity Re-ranking

**Problem**: Top-10 by pure score might be 10 very similar movies (e.g., all Marvel sequels).

**Solution**: Maximal Marginal Relevance (MMR). It re-ranks candidates to balance relevance and diversity:

```
MMR(movie) = lambda * score(movie) - (1 - lambda) * max_similarity_to_already_selected
```

Algorithm:

1. Take top 5\*N candidates by score (e.g., 50 for N=10)
2. Pick the highest-scored movie first
3. For each next pick: penalize candidates that are too similar to movies already picked
4. `lambda = 0.5` means equal weight to relevance and diversity

**Effect**: The final list has variety — different genres, directors, eras — while still being relevant.

---

## Step 7: Web App — User Profile Vector

The web app (`main.py`) goes beyond single-movie recommendations. It builds a **user taste profile** from swipe history.

### Cold Start (< 3 likes)

Random movies, weighted by popularity (popular movies shown more often).

### After 3+ likes — Profile-Based Recommendations

The system constructs a **user profile vector** in the same feature space as movies:

```
profile = sum(weight_i * movie_vector_i) for each swiped movie
```

Feedback weights depend on swipe direction AND source:

- Like a random movie: +1.0 (mild positive — user wasn't guided)
- Dislike a random movie: -0.3 (mild negative)
- Like a model recommendation: +2.5 (strong positive — model was right)
- Dislike a model recommendation: -1.2 (strong negative — model was wrong)
- Like from recommendation list: +3.0 (strongest — deliberate choice)
- Dislike from recommendation list: -1.5

**Why different weights?** Liking a movie the model suggested validates the model's direction (strong signal). Liking a random movie is weaker because it wasn't targeted. Dislikes are weaker than likes because people dislike for many reasons (already seen, not in the mood, etc.).

The profile vector is L2-normalized, then cosine similarity is computed against all movies. Same alpha blending with weighted rating, same MMR re-ranking.

### Additional mechanisms:

- **Exploration rate (15%)**: Even after 3 likes, 15% of movies shown are random. Prevents filter bubble.
- **Dislike penalty**: Movies very similar to disliked ones (cosine > 0.7) get their score halved.

---

## Evaluation

The system is evaluated with these metrics (using genre overlap as proxy for relevance):

| Metric                   | What it measures                                                                       |
| ------------------------ | -------------------------------------------------------------------------------------- |
| **Precision@K**          | Fraction of top-K recs sharing at least one genre with the query movie                 |
| **NDCG@K**               | Whether recs with more genre overlap appear higher in the list                         |
| **Coverage**             | What fraction of the catalog ever gets recommended                                     |
| **Intra-List Diversity** | How different the recommended movies are from each other (1 - avg pairwise similarity) |
| **Novelty**              | Whether the system recommends lesser-known movies (inverse log popularity)             |
| **Serendipity**          | Relevant AND unexpected recommendations (relevant \* (1 - popularity))                 |
| **MRR**                  | How quickly a relevant movie appears in the list (1/rank of first relevant)            |

A **grid search** can tune the field weights (genres, keywords, director, collection) by optimizing 0.5 _ Precision@K + 0.5 _ NDCG@K.

---

## Summary of the Pipeline

```
Raw CSVs
  --> make_dataset (clean, parse, merge)
    --> build_features (per-field text cleaning)
      --> fit vectorizers (TF-IDF / Count per field, weighted, stacked)
        --> cosine similarity matrix (all pairs)
          --> recommend: score = similarity + quality, then MMR diversity
```

Web app adds:

```
User swipes --> weighted profile vector --> cosine vs all movies --> score + MMR --> next movie
                (with 15% random exploration and dislike penalties)
```
