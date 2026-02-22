# Content-Based Movie Recommender System - Design

## Overview

Mini-project 1 for ISA course. Build a content-based movie recommender using The Movies Dataset from Kaggle, with cookiecutter-data-science project structure.

## Approach

**Metadata Soup + TF-IDF + Cosine Similarity + Weighted Rating Pre-filter**

Combine genres, keywords, top 3 cast, director, and plot overview into a single "metadata soup" string per movie. Apply TF-IDF vectorization, compute cosine similarity, and filter results by weighted rating (IMDB formula).

## Data Flow

```
Raw CSVs → Load & Merge → Clean & Preprocess → Feature Engineering → TF-IDF Vectorization → Cosine Similarity Matrix → Recommendation Engine
                                                                                                         ↑
                                                                              Weighted Rating Filter ─────┘
```

**Input files**: `movies_metadata.csv`, `credits.csv`, `keywords.csv`, `links_small.csv`, `ratings_small.csv`

## Deliverables

### 1.1 EDA and Data Preprocessing (12 points)

**A) EDA (4 pts)**: Dataset overview (shape, dtypes, missing values), rating distributions, vote count analysis, genre frequency analysis, top cast/directors, word cloud from overviews, correlation analysis.

**B) Data Preparation (4 pts)**: Parse JSON columns (genres, keywords, cast, crew), handle missing values, merge datasets on movie ID, extract top 3 cast + director, build metadata soup string, text cleaning (lowercase, remove name spaces).

**C) Iterative Justification (4 pts)**:
- Iteration 1: Overview-only TF-IDF recommender
- Iteration 2: Full metadata soup (genres + keywords + cast + crew + overview)
- Iteration 3: Add weighted rating pre-filter (IMDB formula)
- Compare results qualitatively at each step

### 1.2 Modeling and Evaluation (8 points)

**A) Modeling (4 pts)**: TF-IDF Vectorizer on metadata soup, cosine similarity matrix, weighted rating calculation (`WR = (v/(v+m)) * R + (m/(v+m)) * C`), recommendation function returning top-N similar movies.

**B) Evaluation (4 pts)**: Qualitative evaluation (recommendations for known movies), quantitative (Precision@K using genre overlap), comparison across iterations, similarity distribution visualization.

## Tech Stack

- Python 3.10+
- pandas, numpy, scikit-learn (TF-IDF, cosine_similarity)
- matplotlib, seaborn, wordcloud (visualization)
- ast (JSON parsing)
- cookiecutter-data-science project template
