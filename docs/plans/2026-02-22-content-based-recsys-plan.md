# Content-Based Movie Recommender - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a content-based movie recommender system with metadata soup + TF-IDF + cosine similarity + weighted rating pre-filter, structured as a cookiecutter-data-science project.

**Architecture:** Cookiecutter-data-science project with Jupyter notebook as the main deliverable. Raw CSVs flow through loading, merging, cleaning, feature engineering (metadata soup), TF-IDF vectorization, cosine similarity computation, and a weighted rating filter to produce ranked recommendations.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, matplotlib, seaborn, wordcloud, ast, cookiecutter

---

### Task 1: Environment Setup

**Files:**
- Create: `requirements.txt`

**Step 1: Install cookiecutter and generate project structure**

```bash
pip install cookiecutter
```

Then generate the project. Use cookiecutter-data-science v2. The project should be created IN the current repo directory. Since we already have a git repo, we need to generate into a temp dir and move files.

```bash
cd /tmp
cookiecutter https://github.com/drivendata/cookiecutter-data-science --no-input \
  project_name="ISA Movie Recommendation" \
  repo_name="isa-movie-recommendation" \
  author_name="riso" \
  description="Content-based movie recommender system for ISA course" \
  python_version_number="3.12"
```

Then copy the generated structure into our repo:
```bash
cp -r /tmp/isa-movie-recommendation/* /mnt/c/Users/risko/Desktop/ISA-Movie-Recommendation/
cp -r /tmp/isa-movie-recommendation/.* /mnt/c/Users/risko/Desktop/ISA-Movie-Recommendation/ 2>/dev/null || true
```

**Step 2: Create requirements.txt with needed packages**

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
wordcloud>=1.9
jupyter>=1.0
notebook>=7.0
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: scaffold cookiecutter-data-science project structure"
```

---

### Task 2: Data Loading - Place Dataset Files

**Files:**
- Verify: `data/raw/movies_metadata.csv`
- Verify: `data/raw/credits.csv`
- Verify: `data/raw/keywords.csv`
- Verify: `data/raw/links_small.csv`
- Verify: `data/raw/ratings_small.csv`

**Step 1: Verify the user has downloaded and placed the CSVs**

The user must download from https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset and place:
- `movies_metadata.csv` → `data/raw/`
- `credits.csv` → `data/raw/`
- `keywords.csv` → `data/raw/`
- `links_small.csv` → `data/raw/`
- `ratings_small.csv` → `data/raw/`

**Step 2: Quick validation that files exist and have expected columns**

```python
import pandas as pd
for f in ['movies_metadata.csv', 'credits.csv', 'keywords.csv', 'links_small.csv', 'ratings_small.csv']:
    df = pd.read_csv(f'data/raw/{f}', nrows=2)
    print(f"{f}: {df.columns.tolist()}")
```

---

### Task 3: Notebook - EDA Section (1.1A)

**Files:**
- Create: `notebooks/01_eda_and_preprocessing.ipynb`

**Step 1: Create notebook with imports and data loading cells**

Cell 1 (markdown): Title and description
Cell 2 (code): Imports (pandas, numpy, matplotlib, seaborn, wordcloud, ast, warnings)
Cell 3 (code): Load all 5 CSVs with `pd.read_csv()`
Cell 4 (code): `df.shape`, `df.dtypes`, `df.info()` for each dataset
Cell 5 (code): `df.describe()` for movies_metadata
Cell 6 (code): Missing values heatmap using seaborn
Cell 7 (code): Rating distribution histogram (`vote_average`)
Cell 8 (code): Vote count distribution (log scale)
Cell 9 (code): Parse genres column with `ast.literal_eval`, count genre frequencies, bar chart
Cell 10 (code): Parse cast/crew, find most frequent actors and directors, bar charts
Cell 11 (code): Word cloud from movie overviews
Cell 12 (code): Correlation heatmap of numeric features (budget, revenue, popularity, vote_average, vote_count)
Cell 13 (markdown): Summary of EDA findings and quest establishment

**Step 2: Run all cells to verify outputs**

**Step 3: Commit**

```bash
git add notebooks/01_eda_and_preprocessing.ipynb
git commit -m "feat: add EDA section with visualizations"
```

---

### Task 4: Notebook - Data Preprocessing (1.1B)

**Files:**
- Modify: `notebooks/01_eda_and_preprocessing.ipynb`

**Step 1: Add preprocessing cells after EDA section**

Cell 14 (markdown): "## Data Preprocessing"
Cell 15 (code): Clean movies_metadata - fix bad IDs (some rows have date strings in id column), convert id to int, drop rows with bad data
Cell 16 (code): Parse JSON-like string columns using `ast.literal_eval` for genres, keywords (from keywords.csv), cast, crew (from credits.csv)
Cell 17 (code): Extract features:
  - genres: list of genre names
  - keywords: list of keyword names
  - cast: top 3 actor names
  - director: director name from crew
Cell 18 (code): Merge all datasets on movie id (movies_metadata + credits + keywords)
Cell 19 (code): Handle missing values - fill NaN overviews with '', drop rows with no usable features
Cell 20 (code): Text cleaning function:
  - Lowercase all text
  - Remove spaces from names (e.g., "Johnny Depp" → "johnnydepp") so they become single tokens
  - Apply to cast, director, keywords, genres
Cell 21 (code): Build "metadata soup" - concatenate: overview + ' ' + genres + ' ' + keywords + ' ' + cast + ' ' + director
Cell 22 (code): Show sample soup strings, verify they look correct
Cell 23 (code): Save processed dataframe to `data/processed/movies_processed.csv`
Cell 24 (markdown): Summary of preprocessing decisions and justification

**Step 2: Run all cells**

**Step 3: Commit**

```bash
git add notebooks/01_eda_and_preprocessing.ipynb data/processed/
git commit -m "feat: add data preprocessing pipeline"
```

---

### Task 5: Notebook - Iterative Modeling (1.1C + 1.2A)

**Files:**
- Modify: `notebooks/01_eda_and_preprocessing.ipynb`

**Step 1: Add Iteration 1 - Overview-only recommender**

Cell 25 (markdown): "## Modeling - Iterative Approach"
Cell 26 (markdown): "### Iteration 1: Overview-Only TF-IDF"
Cell 27 (code): TF-IDF on overview column only, cosine similarity, recommendation function
Cell 28 (code): Test with example movies: "The Dark Knight", "Toy Story", "The Godfather" - show top 10 recommendations
Cell 29 (markdown): Analysis of Iteration 1 results

**Step 2: Add Iteration 2 - Full metadata soup**

Cell 30 (markdown): "### Iteration 2: Metadata Soup TF-IDF"
Cell 31 (code): TF-IDF on full metadata soup, cosine similarity, recommendation function
Cell 32 (code): Test with same movies, compare to Iteration 1
Cell 33 (markdown): Analysis - how metadata soup improved results

**Step 3: Add Iteration 3 - Weighted rating pre-filter**

Cell 34 (markdown): "### Iteration 3: Metadata Soup + Weighted Rating Filter"
Cell 35 (code): Calculate weighted rating using IMDB formula:
```python
C = df['vote_average'].mean()
m = df['vote_count'].quantile(0.60)  # minimum votes to be considered
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(v+m) * C)
```
Cell 36 (code): Final recommendation function that:
  1. Gets top-N*3 similar movies by cosine similarity
  2. Filters to movies with vote_count >= m
  3. Sorts by weighted rating
  4. Returns top-N
Cell 37 (code): Test with same movies, compare all 3 iterations side by side
Cell 38 (markdown): Analysis - improvement from adding quality filter

**Step 4: Commit**

```bash
git add notebooks/01_eda_and_preprocessing.ipynb
git commit -m "feat: add iterative modeling with 3 approaches"
```

---

### Task 6: Notebook - Evaluation (1.2B)

**Files:**
- Modify: `notebooks/01_eda_and_preprocessing.ipynb`

**Step 1: Add evaluation cells**

Cell 39 (markdown): "## Evaluation"
Cell 40 (code): Qualitative evaluation - table showing recommendations for 5 well-known movies across all 3 iterations
Cell 41 (code): Quantitative - Precision@K using genre overlap:
```python
def precision_at_k(movie_title, k=10):
    """Fraction of recommended movies sharing at least one genre with input"""
    input_genres = set(get_genres(movie_title))
    recs = get_recommendations(movie_title, k)
    hits = sum(1 for r in recs if input_genres & set(get_genres(r)))
    return hits / k
```
Cell 42 (code): Compute Precision@10 for a sample of 50 movies across all 3 iterations, bar chart comparison
Cell 43 (code): Similarity score distribution histogram for the final model
Cell 44 (code): Heatmap/table comparing iterations on evaluation metrics
Cell 45 (markdown): "## Conclusion" - summary of results, best model justification, limitations, future work

**Step 2: Run all cells to verify**

**Step 3: Commit**

```bash
git add notebooks/01_eda_and_preprocessing.ipynb
git commit -m "feat: add evaluation section with metrics and comparisons"
```

---

### Task 7: Data Pipeline Module

**Files:**
- Create: `src/data/make_dataset.py`
- Create: `src/features/build_features.py`
- Create: `src/models/predict_model.py`

**Step 1: Create data loading module**

`src/data/make_dataset.py`: Functions to load raw CSVs, clean, merge, and save processed data.

**Step 2: Create feature engineering module**

`src/features/build_features.py`: Functions for parsing JSON columns, text cleaning, building metadata soup.

**Step 3: Create prediction module**

`src/models/predict_model.py`: Functions for TF-IDF vectorization, cosine similarity, weighted rating, and get_recommendations().

**Step 4: Commit**

```bash
git add src/
git commit -m "feat: add reusable data pipeline modules"
```

---

### Task 8: Final Cleanup and Verification

**Step 1: Restart kernel and run all cells in notebook**

Ensure the notebook runs end-to-end without errors.

**Step 2: Verify project structure matches cookiecutter-data-science**

```
├── data/
│   ├── raw/          (CSVs from Kaggle)
│   └── processed/    (cleaned data)
├── docs/plans/
├── notebooks/
│   └── 01_eda_and_preprocessing.ipynb
├── src/
│   ├── data/make_dataset.py
│   ├── features/build_features.py
│   └── models/predict_model.py
├── models/           (saved similarity matrix if needed)
├── requirements.txt
└── README.md
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
```
