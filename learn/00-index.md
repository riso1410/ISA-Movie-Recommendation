# MovieMatch Recommendation System — Learning Guide

A comprehensive guide to understanding the content-based movie recommendation system, from raw data to live recommendations. Written for readers with limited ML/NLP background.

## Reading Order

Each document builds on the previous one. Read sequentially for the best experience.

| #   | Document                                                 | What You'll Learn                                                                                                                |
| --- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 01  | [Data Preparation](01-data-preparation.md)               | Raw Kaggle CSVs → cleaned, merged DataFrame. JSON parsing, ID cleaning, merging strategies, poster scraping.                     |
| 02  | [TF-IDF & Vectorization](02-tfidf-and-vectorization.md)  | How text becomes numbers. Bag of Words → TF-IDF math → multi-vectorizer architecture with per-field weights and sparse matrices. |
| 03  | [Similarity & Scoring](03-similarity-and-scoring.md)     | Cosine similarity (with 2D intuition), IMDB weighted rating formula, combined scoring, MMR diversity re-ranking.                 |
| 04  | [Training & Persistence](04-training-and-persistence.md) | What "training" means for this system (not gradient descent), the fit pipeline, pickle serialization, model loading at startup.  |
| 05  | [User Interaction & App](05-user-interaction-and-app.md) | Swipe UX, session state, cold start vs warm phase, recommendation aggregation from multiple liked movies, API endpoints.         |
| 06  | [Evaluation Metrics](06-evaluation-metrics.md)           | Precision@K, NDCG@K, Coverage, Intra-List Diversity, Novelty, grid search weight optimization.                                   |

## System Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  5 Raw CSVs │────▶│ make_dataset │────▶│ build_features  │────▶│  fit() model │
│  (Kaggle)   │     │  clean+merge │     │ per-field text  │     │  TF-IDF +    │
│  Doc: 01    │     │  Doc: 01     │     │  Doc: 01, 02    │     │  cosine sim  │
└─────────────┘     └──────────────┘     └─────────────────┘     │  Doc: 02, 03 │
                                                                  └──────┬───────┘
                                                                         │
                                              pickle save/load           │
                                              Doc: 04                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Browser   │◀───▶│  FastAPI     │◀───▶│ recommend()     │◀────│ recommender  │
│  Swipe UI   │     │  endpoints   │     │ scoring + MMR   │     │    .pkl      │
│  Doc: 05    │     │  Doc: 05     │     │ Doc: 03, 05     │     │  Doc: 04     │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
                                                                         │
                                                                         ▼
                                                                  ┌──────────────┐
                                                                  │  evaluate()  │
                                                                  │  P@K, NDCG,  │
                                                                  │  Coverage... │
                                                                  │  Doc: 06     │
                                                                  └──────────────┘
```

## Key Concepts Quick Reference

| Concept              | One-line explanation                                 | Document |
| -------------------- | ---------------------------------------------------- | -------- |
| TF-IDF               | Words weighted by uniqueness across documents        | 02       |
| Cosine Similarity    | Angle between feature vectors = movie similarity     | 03       |
| IMDB Weighted Rating | Bayesian blend of movie rating and global average    | 03       |
| MMR                  | Re-ranks results to maximize diversity               | 03       |
| Sparse Matrix        | Memory-efficient storage for mostly-zero data        | 02       |
| Cold Start           | Random popular movies shown before enough likes      | 05       |
| Aggregation          | Combining recommendations from multiple liked movies | 05       |
| Precision@K          | Fraction of top-K recommendations that are relevant  | 06       |
| NDCG@K               | Measures if best results are ranked highest          | 06       |
| Coverage             | Fraction of catalog ever recommended                 | 06       |
