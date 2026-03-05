# MovieMatch Recommendation System — Learning Guide

A comprehensive guide to understanding the content-based movie recommendation system, from raw data to live recommendations. Written for readers with limited ML/NLP background.

## Reading Order

Each document builds on the previous one. Read sequentially for the best experience.

| #   | Document                                                 | What You'll Learn                                                                                                                |
| --- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 01  | [Data Preparation](01-data-preparation.md)               | Raw Kaggle CSVs → cleaned, merged DataFrame. JSON parsing, ID cleaning, merging strategies, poster scraping.                     |
| 02  | [TF-IDF & Vectorization](02-tfidf-and-vectorization.md)  | How text becomes numbers. Bag of Words → TF-IDF math → multi-vectorizer architecture with per-field weights and sparse matrices. |
| 03  | [Similarity & Scoring](03-similarity-and-scoring.md)     | Cosine similarity, IMDB weighted rating, combined scoring, MMR diversity, user profile vector, soft penalty, exploration.        |
| 04  | [Training & Persistence](04-training-and-persistence.md) | What "training" means for this system (not gradient descent), the fit pipeline, pickle serialization, model loading at startup.  |
| 05  | [User Interaction & App](05-user-interaction-and-app.md) | Swipe UX, session state, three-phase movie selection, profile vector engine, rec-list feedback, API endpoints.                   |
| 06  | [Evaluation Metrics](06-evaluation-metrics.md)           | Precision@K, NDCG@K, MRR, Serendipity, Coverage, Intra-List Diversity, Novelty, Per-Genre Precision, grid search.                |

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
│   Browser   │◀───▶│  FastAPI     │◀───▶│ profile vector  │◀────│ recommender  │
│  Swipe UI + │     │  endpoints   │     │ + scoring + MMR │     │    .pkl      │
│  Rec List   │     │  Doc: 05     │     │ Doc: 03, 05     │     │  Doc: 04     │
│  Doc: 05    │     │              │     │                 │     │              │
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

| Concept              | One-line explanation                                | Document |
| -------------------- | --------------------------------------------------- | -------- |
| TF-IDF               | Words weighted by uniqueness across documents       | 02       |
| Cosine Similarity    | Angle between feature vectors = movie similarity    | 03       |
| IMDB Weighted Rating | Bayesian blend of movie rating and global average   | 03       |
| MMR                  | Re-ranks results to maximize diversity              | 03       |
| Sparse Matrix        | Memory-efficient storage for mostly-zero data       | 02       |
| Cold Start           | Random popular movies shown before enough likes     | 05       |
| User Profile Vector  | Weighted sum of all feedback → single taste vector  | 03, 05   |
| Feedback Weights     | Source-aware weights: rec_list > model > random     | 03, 05   |
| Soft Penalty         | Halve scores for movies similar to disliked ones    | 03       |
| Exploration Rate     | 15% epsilon-greedy random to break echo chambers    | 03, 05   |
| Rec-List Feedback    | Like/dislike on sidebar cards (highest weight)      | 05       |
| Precision@K          | Fraction of top-K recommendations that are relevant | 06       |
| NDCG@K               | Measures if best results are ranked highest         | 06       |
| MRR                  | Reciprocal rank of first relevant recommendation    | 06       |
| Serendipity@K        | Unexpectedness times relevance of recommendations   | 06       |
| Per-Genre Precision  | Precision@K broken down by individual genre         | 06       |
| Coverage             | Fraction of catalog ever recommended                | 06       |
