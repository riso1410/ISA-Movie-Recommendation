# 04 - Training, Persistence, and Loading the Recommender Model

This document explains how the MovieMatch recommendation model is trained, saved to
disk, and loaded back into memory when the web application starts. Every concept is
explained from scratch -- no prior machine-learning knowledge is assumed.

---

## Table of Contents

1. [What Does "Training" Mean Here?](#1-what-does-training-mean-here)
2. [The Training Pipeline Step by Step](#2-the-training-pipeline-step-by-step)
3. [Model Persistence with Pickle](#3-model-persistence-with-pickle)
4. [Model Loading at App Startup](#4-model-loading-at-app-startup)
5. [ContentBasedRecommender Class Anatomy](#5-contentbasedrecommender-class-anatomy)
6. [Hot-Swapping and Runtime Operations](#6-hot-swapping-and-runtime-operations)
7. [Limitations of This Approach](#7-limitations-of-this-approach)

---

## 1. What Does "Training" Mean Here?

### The word "training" can be misleading

When most people hear "training a model" they think of deep learning: a neural
network is shown millions of labeled examples, it computes a loss function, and
gradient descent slowly adjusts millions of weight parameters over many iterations
(epochs). That is **not** what happens here.

This recommender is **not**:

- A neural network
- A classifier that predicts labels
- A system that uses gradient descent
- A model that learns from user ratings or click data

### What "training" actually means for this system

"Training" in this project means three concrete things:

```
 Step 1:  Build a vocabulary
           - Scan every movie's text fields (overview, genres, keywords, cast, director, etc.)
           - Decide which words/tokens matter and assign each one a numeric ID
           - For TF-IDF fields: also compute IDF weights (how rare each word is)

 Step 2:  Compute feature vectors
           - Convert each movie's text into a numeric vector using the vocabulary
           - Each movie becomes a row in a large sparse matrix
           - Combine all per-field matrices into one big feature matrix

 Step 3:  Compute the similarity matrix
           - Compare every movie to every other movie using cosine similarity
           - Result: an N x N matrix where entry [i][j] = how similar movie i is to movie j
```

### Supervised vs. unsupervised

Traditional recommendation systems like collaborative filtering learn from **user
behavior** (ratings, clicks, purchases). They need labeled data: "User A gave Movie X
a rating of 4.5."

This system is **unsupervised** -- it never sees user ratings during training. Instead
it discovers structure from the **content** (metadata) of the movies themselves:

```
  Supervised (collaborative filtering):       Unsupervised (content-based):
  ----------------------------------------    ----------------------------------------
  Input:  User-item rating matrix              Input:  Movie metadata text
  Learns: Which users have similar tastes      Learns: Which movies have similar content
  Needs:  Lots of user interaction data        Needs:  Just the movie descriptions
  Fails:  New users (cold start)               Fails:  New movies (need refit)
```

### What artifacts are produced

After training completes, the model holds these artifacts in memory:

```
  Artifact                  What it is                                 Example size
  ---------------------------------------------------------------------------------
  vectorizers (dict)        Fitted vocabulary + IDF weights             8 vectorizers
                            per field (overview, genres, etc.)          (one per field)

  feature_matrix (sparse)   Combined TF-IDF/Count matrix               ~9,000 rows x
                            One row per movie, one column per           ~20,000 columns
                            token across all fields

  cosine_sim (dense)        N x N similarity matrix                    ~9,000 x 9,000
                            cosine_sim[i][j] = similarity              = ~81 million
                            between movie i and movie j                float values

  smd (DataFrame)           The full movie metadata table               ~9,000 rows x
                            (titles, genres, ratings, etc.)             ~20 columns

  indices (Series)          title -> row number mapping                 ~9,000 entries
                            for O(1) lookup

  weighted_rating (column)  IMDB-style quality score per movie         1 float per movie
  wr_norm (column)          Normalized 0-1 version of above            1 float per movie
```

---

## 2. The Training Pipeline Step by Step

The entire training process is orchestrated by `src/models/train_model.py`:

```python
def train(raw_dir='data/raw', model_dir='models'):
    smd = make_dataset(raw_dir)       # Step 1: Load and clean data
    smd = build_features(smd)          # Step 2: Create text columns for vectorizers
    recommender = build_recommender(smd)  # Step 3: Instantiate + fit recommender
    # Step 4: Save to disk with pickle
    with open(model_dir / 'recommender.pkl', 'wb') as f:
        pickle.dump(recommender, f)
```

Let's walk through each step.

### Step 1: `make_dataset()` -- Load and clean the raw data

**File:** `src/data/make_dataset.py`

This function loads five raw CSV files from the Kaggle "The Movies Dataset":

```
  movies_metadata.csv   -- titles, overviews, genres, vote counts, etc.
  credits.csv           -- cast and crew (JSON strings)
  keywords.csv          -- keyword tags (JSON strings)
  links_small.csv       -- maps MovieLens IDs to TMDB IDs (our subset filter)
  ratings_small.csv     -- user ratings (loaded but not used for training)
```

What it does:

```
  1. Load all 5 CSVs into DataFrames
  2. Remove movies with non-numeric IDs (data quality issue in the dataset)
  3. Filter to only movies in links_small (~9,000 movies from ~45,000)
  4. Parse the JSON strings in credits.csv to extract:
     - Top 5 cast members (by billing order)
     - Director name
  5. Parse the JSON strings in keywords.csv to extract keyword names
  6. Parse genres from JSON to a list of genre names
  7. Merge everything together on movie ID
  8. Derive new columns: decade (e.g., "decade_1990s"), language (e.g., "lang_en"),
     collection (e.g., "Toy Story Collection")
  9. Save to data/processed/movies_processed.csv
```

The output is a single DataFrame (`smd`) with approximately 9,000 rows and columns
like: `id, title, overview, genres, keywords, cast, director, vote_average,
vote_count, popularity, decade, language, collection`.

### Step 2: `build_features()` -- Prepare text columns for vectorization

**File:** `src/features/build_features.py`

The vectorizers need plain text strings. This function creates clean text columns
from the structured data:

```
  Source column       -->  Feature column      Example transformation
  ---------------------------------------------------------------------------
  overview            -->  overview_clean       "A cowboy toy..." -> "a cowboy toy..."
  genres (list)       -->  genres_str           ["Animation", "Comedy"] -> "animation comedy"
  keywords (list)     -->  keywords_str         ["toys", "friendship"] -> "toys friendship"
  cast (list)         -->  cast_str             ["Tom Hanks", "Tim Allen"] -> "tomhanks timallen"
  director (string)   -->  director_str         "John Lasseter" -> "johnlasseter"
  decade              -->  decade               "decade_1990s" (already a token)
  language            -->  language              "lang_en" (already a token)
  collection          -->  collection            "toystorycollection"
```

**Why remove spaces from names?** Consider "Tom Hanks". If we leave it as two words,
TF-IDF treats "Tom" and "Hanks" as separate tokens. A movie with "Tom Cruise" would
partially match on "Tom". By merging to "tomhanks", the actor becomes a single unique
token, ensuring only exact actor matches contribute to similarity.

### Step 3: `build_recommender()` -- Create and fit the recommender

**File:** `src/models/predict_model.py`

This is a convenience function that does two things:

```python
def build_recommender(smd, weights=None):
    recommender = ContentBasedRecommender(smd, weights=weights)  # __init__
    recommender.fit()                                             # fit
    return recommender
```

#### What happens inside `__init__`:

```python
def __init__(self, smd, weights=None):
    self.smd = smd.reset_index(drop=True)           # Store the movie DataFrame
    indices = pd.Series(self.smd.index, index=self.smd['title'])
    self.indices = indices[~indices.index.duplicated(keep='first')]  # title -> row#
    self.weights = weights or DEFAULT_WEIGHTS.copy()  # field importance weights
    self.vectorizers = {}                              # will hold fitted vectorizers
    self.feature_matrix = None                         # will hold combined sparse matrix
    self.cosine_sim = None                             # will hold N x N similarity

    self.C = self.smd['vote_average'].mean()           # mean rating across all movies
    self.m = self.smd['vote_count'].quantile(0.60)     # 60th percentile vote count
    self._compute_weighted_ratings()                    # compute IMDB-formula scores
```

The **indices** Series is a lookup table:

```
  "Toy Story"        -->  0
  "Jumanji"          -->  1
  "Heat"             -->  2
  "The Dark Knight"  -->  456
  ...
```

If two movies have the same title, only the first one is kept (duplicate handling).

The **weighted rating** uses the IMDB formula:

```
  WR = (v / (v + m)) * R + (m / (v + m)) * C

  where:
    v = number of votes for this movie
    m = minimum votes required (60th percentile, about ~20 votes)
    R = average rating for this movie
    C = mean rating across ALL movies (~6.0)
```

This formula shrinks ratings toward the global mean for movies with few votes.
A movie rated 9.0 by only 3 people gets pulled down toward 6.0. A movie rated 8.5
by 10,000 people stays close to 8.5.

#### What happens inside `fit()`:

This is where the real computation happens:

```python
def fit(self):
    matrices = []

    for field, (col, vec_cls, vec_kwargs) in FIELD_CONFIG.items():
        weight = self.weights.get(field, 0.0)
        if weight == 0.0:
            continue

        texts = self.smd[col].fillna('').astype(str)
        vectorizer = vec_cls(**vec_kwargs)
        matrix = vectorizer.fit_transform(texts)   # <-- Build vocabulary + transform

        if weight != 1.0:
            matrix = matrix * weight                # <-- Scale by field importance

        self.vectorizers[field] = vectorizer
        matrices.append(matrix)

    self.feature_matrix = hstack(matrices, format='csr')      # <-- Combine all fields
    self.cosine_sim = cosine_similarity(self.feature_matrix)   # <-- N x N similarity
    return self
```

Let's visualize what this loop does for each field:

```
  Field Configuration (FIELD_CONFIG):
  -----------------------------------------------------------------------
  Field       Column           Vectorizer        Weight   Max Features
  -----------------------------------------------------------------------
  overview    overview_clean   TfidfVectorizer   1.0      15,000
  genres      genres_str       CountVectorizer   1.5      --
  keywords    keywords_str     TfidfVectorizer   1.2      5,000
  cast        cast_str         CountVectorizer   1.0      --
  director    director_str     CountVectorizer   2.0      --
  decade      decade           CountVectorizer   0.3      --
  language    language          CountVectorizer   0.5      --
  collection  collection       CountVectorizer   1.5      --
  -----------------------------------------------------------------------
```

**TfidfVectorizer vs CountVectorizer:**

- `CountVectorizer` counts how many times each token appears. For binary fields like
  genres ("animation comedy"), each token appears 0 or 1 times.
- `TfidfVectorizer` also weights by Inverse Document Frequency (IDF) -- tokens that
  appear in fewer documents get higher weight. This is important for overview text
  where common words like "movie" should matter less than distinctive words like
  "cyberpunk".

**The weight multiplication:** After vectorizing, each field's matrix is multiplied
by its weight. Director has weight 2.0, meaning director similarity counts double.
Decade has weight 0.3, so it's a mild signal. This is how the system encodes domain
knowledge about which metadata fields matter most.

**Horizontal stacking (`hstack`):**

```
  overview matrix      genres matrix    keywords matrix   ...   collection matrix
  [9000 x 15000]   +  [9000 x ~20]  + [9000 x 5000]   + ... + [9000 x ~200]
  \_______________________________________________ _____________________________/
                                                  |
                                         hstack (horizontal concatenation)
                                                  |
                                                  v
                                      feature_matrix [9000 x ~20,200]
                                      (one combined sparse matrix)
```

Each movie is now a single row of ~20,000 numbers. Most of these numbers are zero
(the matrix is "sparse" -- stored efficiently so zeros don't use memory).

**Cosine similarity computation:**

```
  cosine_similarity(feature_matrix)

  Input:   Sparse matrix  [9,000 x 20,200]
  Output:  Dense matrix   [9,000 x 9,000]

  Each cell [i][j] = cosine of the angle between movie i's vector and movie j's vector
                   = dot(movie_i, movie_j) / (||movie_i|| * ||movie_j||)
                   = a value between 0.0 (completely different) and 1.0 (identical)
```

**Why `fit()` takes ~15 seconds:** The cosine similarity step must compare every pair
of movies. With ~9,000 movies, that's 9,000 x 9,000 = 81 million similarity values
to compute. Each computation involves a dot product across ~20,000 dimensions. The
scikit-learn implementation uses optimized BLAS routines, but it's still O(N^2) and
produces a dense matrix that consumes ~620 MB of RAM (81M floats x 8 bytes each).

Here's a diagram of the complete training data flow:

```
  5 Raw CSV Files (Kaggle)
        |
        v
  make_dataset()
        |  - load CSVs
        |  - clean IDs, filter to ~9,000 movies
        |  - parse JSON columns (cast, keywords, genres)
        |  - merge all tables on movie ID
        |  - derive decade, language, collection
        v
  Single DataFrame (smd): ~9,000 rows x ~20 columns
        |
        v
  build_features()
        |  - create overview_clean, genres_str, keywords_str,
        |    cast_str, director_str columns
        |  - remove spaces from names for single-token matching
        v
  DataFrame with text feature columns added
        |
        v
  ContentBasedRecommender.__init__()
        |  - build title -> row index mapping
        |  - compute IMDB weighted ratings
        |
        v
  ContentBasedRecommender.fit()
        |  - for each field: fit vectorizer + transform text -> sparse matrix
        |  - multiply each matrix by field weight
        |  - hstack all matrices into one combined feature matrix
        |  - compute N x N cosine similarity matrix
        v
  Fully fitted recommender object
        |
        v
  pickle.dump()  -->  models/recommender.pkl  (~150-300 MB on disk)
```

---

## 3. Model Persistence with Pickle

### What is pickle?

Pickle is Python's built-in serialization library. It converts a live Python object
in memory into a stream of bytes that can be saved to a file and later restored.

Think of it like freezing food:

```
  Live Python object (in RAM)
        |
        |  pickle.dump(obj, file)    <-- "freeze" the object
        v
  Bytes on disk (recommender.pkl)
        |
        |  pickle.load(file)         <-- "thaw" the object
        v
  Live Python object (in RAM, identical to original)
```

### What exactly gets saved?

When we call `pickle.dump(recommender, f)`, pickle traverses the entire object graph
and serializes everything reachable from the `recommender` object:

```
  recommender.pkl contains:
  +------------------------------------------------------------------+
  |  ContentBasedRecommender                                         |
  |  +------------------------------------------------------------+  |
  |  | smd (DataFrame)           ~9,000 rows of movie metadata    |  |
  |  | cosine_sim (ndarray)      9,000 x 9,000 float64 matrix    |  |
  |  | feature_matrix (sparse)   9,000 x ~20,200 sparse matrix   |  |
  |  | vectorizers (dict)        8 fitted vectorizer objects,     |  |
  |  |                           each storing:                    |  |
  |  |                           - vocabulary (word -> column#)   |  |
  |  |                           - idf weights (for TF-IDF ones)  |  |
  |  | indices (Series)          title -> row# mapping            |  |
  |  | weights (dict)            field weight configuration       |  |
  |  | C (float)                 mean vote_average                |  |
  |  | m (float)                 60th percentile vote_count       |  |
  |  +------------------------------------------------------------+  |
  +------------------------------------------------------------------+
```

### File size implications

The dominant contributor to file size is the `cosine_sim` matrix:

```
  9,000 x 9,000 x 8 bytes (float64) = ~617 MB (uncompressed in memory)
```

Pickle applies some compression, and the actual `.pkl` file is typically 150-300 MB
depending on the exact number of movies. The sparse `feature_matrix` is small on disk
because pickle preserves its sparse representation.

### Why pickle and not other formats?

| Format     | Pros                                                                                                        | Cons                                                                                      |
| ---------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **pickle** | Saves arbitrary Python objects. One line of code. Preserves exact class structure including custom methods. | Python-version-sensitive. Not human-readable. Security risk.                              |
| **JSON**   | Human-readable, language-agnostic                                                                           | Cannot serialize NumPy arrays, sparse matrices, or class instances directly               |
| **joblib** | Optimized for large NumPy arrays (better compression)                                                       | Still Python-specific. Slightly more complex API.                                         |
| **ONNX**   | Portable ML model format                                                                                    | Designed for neural networks, not for a custom class with a DataFrame + similarity matrix |
| **HDF5**   | Good for large numeric arrays                                                                               | Would need to save/load each piece separately, plus custom reconstruction logic           |

Pickle is the pragmatic choice for a prototype: one line to save, one line to load,
and it preserves the exact object including all methods.

### Risks and caveats

**Python version sensitivity:** A pickle file created with Python 3.10 may fail to
load on Python 3.12 if the internal representation of any class changed. In practice,
this is rare for NumPy/pandas/scikit-learn objects, but it happens.

**Library version sensitivity:** If you upgrade scikit-learn (e.g., from 1.3 to 1.5)
and the internal structure of `TfidfVectorizer` changed, loading the old pickle will
fail. The fix is to retrain: `uv run python -m src.models.train_model`.

**Security:** Pickle can execute arbitrary code during deserialization. Never load a
pickle file from an untrusted source. In this project the pickle is created locally,
so this is not a concern.

**Not portable across languages:** Unlike ONNX or PMML, a Python pickle cannot be
loaded from Java, JavaScript, or any other language.

---

## 4. Model Loading at App Startup

When the FastAPI web application starts, the `startup()` function in `app/main.py`
runs. It has two code paths:

```
  App starts
      |
      v
  Does models/recommender.pkl exist?
      |                    |
     YES                  NO
      |                    |
      v                    v
  Load from pickle     Fit from scratch
  (fast: ~2-5 sec)     (slow: ~15-20 sec)
      |                    |
      v                    v
  Post-processing      Store in memory
      |
      v
  Ready to serve requests
```

### Path A: Loading from pickle (fast path)

```python
model_path = PROJECT_ROOT / "models" / "recommender.pkl"
if model_path.exists():
    with open(model_path, "rb") as f:
        recommender = pickle.load(f)
```

After loading, three post-processing steps happen:

**1. Reset index and coerce numeric types:**

```python
recommender.smd = recommender.smd.reset_index(drop=True)
for col in ["vote_average", "vote_count", "popularity"]:
    recommender.smd[col] = pd.to_numeric(recommender.smd[col], errors="coerce").fillna(0.0)
```

Why? The pickle was created at a different time, potentially with a different pandas
version. Resetting the index ensures row numbers are 0, 1, 2, ... with no gaps.
Coercing to numeric guards against columns that may have loaded as strings.

**2. Merge poster URLs from the processed CSV:**

```python
csv_poster = processed.drop_duplicates(subset="id").set_index("id")["poster_url"]
recommender.smd["poster_url"] = recommender.smd["id"].map(csv_poster).fillna("")
```

**Why poster URLs need special handling:** Poster URLs come from a separate scraping
step (`src/data/fetch_posters.py`) that hits the TMDB API. These URLs are stored in
`data/processed/movies_processed.csv` but were NOT present when the model was trained
(training only uses metadata for similarity, not images). So after loading the pickled
model, the app maps poster URLs from the CSV onto the model's DataFrame.

**3. Fallback to poster_path:**

```python
if "poster_path" in recommender.smd.columns:
    missing = recommender.smd["poster_url"] == ""
    paths = recommender.smd.loc[missing, "poster_path"].fillna("")
    recommender.smd.loc[missing, "poster_url"] = paths.apply(
        lambda p: f"https://image.tmdb.org/t/p/w500{p}" if p else ""
    )
```

Some movies may not have had their posters scraped yet. If the original metadata
includes a `poster_path` field (a partial TMDB URL like `/abc123.jpg`), the app
constructs the full URL as a fallback.

### Path B: Fitting from scratch (fallback path)

```python
else:
    processed = build_features(processed)
    recommender = ContentBasedRecommender(processed)
    recommender.fit()
    movies_df = processed.reset_index(drop=True)
```

If no pickle file exists, the app builds features from the processed CSV and fits the
recommender in-process. This takes ~15-20 seconds, during which the app cannot serve
requests. This is why pre-training and pickling is preferred for production.

### Visual summary of startup data flow

```
                    movies_processed.csv
                           |
                    +------+------+
                    |             |
                    v             v
            Parse columns    Read poster_url
            (genres, etc.)   column
                    |             |
                    v             |
        recommender.pkl exists?  |
           /          \          |
         YES           NO       |
          |             |        |
          v             v        |
    pickle.load()   build_features()
          |          + fit()     |
          v             |        |
    Recommender obj     v        |
          |        Recommender   |
          v          obj         |
    Reset index,     |           |
    coerce types     |           |
          |          |           |
          v          |           |
    Merge poster  <--+-----------+
    URLs from CSV
          |
          v
    recommender & movies_df
    stored as globals
          |
          v
    "Loaded 9000 movies, recommender ready!"
```

---

## 5. ContentBasedRecommender Class Anatomy

Here is every attribute of the class and what it stores:

```python
class ContentBasedRecommender:
    # --- Set in __init__ ---
    smd: pd.DataFrame
    #   The complete movie metadata DataFrame. ~9,000 rows.
    #   Columns include: id, title, overview, genres, keywords, cast, director,
    #   vote_average, vote_count, popularity, decade, language, collection,
    #   overview_clean, genres_str, keywords_str, cast_str, director_str,
    #   weighted_rating, wr_norm
    #
    #   WHY stored: The recommender needs this to return full movie details
    #   (title, genres, ratings) with recommendations, not just row indices.

    indices: pd.Series
    #   Maps movie title (string) -> row index (integer) in smd.
    #   Example: indices["The Dark Knight"] = 456
    #   Used for O(1) lookup when user asks for recommendations for a title.
    #
    #   DUPLICATE HANDLING: If two movies share the same title (e.g., two
    #   different movies both called "Crash"), only the first one is kept:
    #     indices = indices[~indices.index.duplicated(keep='first')]

    weights: dict[str, float]
    #   Field importance weights. Example: {'overview': 1.0, 'director': 2.0, ...}
    #   Used during fit() to scale each field's contribution to similarity.

    vectorizers: dict[str, object]
    #   Maps field name -> fitted vectorizer object.
    #   Example: vectorizers['overview'] is a fitted TfidfVectorizer
    #   Each vectorizer stores:
    #     - vocabulary_: dict mapping token -> column index
    #     - idf_: array of IDF weights (TfidfVectorizer only)
    #   Not used after fit() in the current code, but useful for debugging
    #   or for transforming new documents.

    feature_matrix: scipy.sparse.csr_matrix
    #   Combined sparse matrix. Shape: [n_movies x total_features].
    #   Created by hstack-ing all per-field matrices.
    #   Not used after cosine_sim is computed, but stored for potential reuse.

    cosine_sim: numpy.ndarray
    #   Dense N x N matrix of pairwise cosine similarities.
    #   cosine_sim[i][j] = similarity between movie at row i and movie at row j.
    #   Values range from 0.0 (no similarity) to 1.0 (identical content).
    #   This is THE core data structure used at recommendation time.

    C: float
    #   Mean vote_average across all movies. Used in IMDB weighted rating formula.

    m: float
    #   60th percentile of vote_count. Minimum votes threshold for weighted rating.

    # --- Computed columns added to smd ---
    # smd['weighted_rating']: IMDB-style weighted rating per movie
    # smd['wr_norm']:         weighted_rating normalized to [0, 1] range
```

### How `recommend()` uses these attributes

When a user asks for recommendations similar to "The Dark Knight":

```
  1. Look up row index:  idx = self.indices["The Dark Knight"]  -->  456

  2. Get similarity row:  sim_scores = self.cosine_sim[456]
     This is a 1D array of 9,000 floats -- similarity to every other movie.

  3. Sort by similarity:  candidate_indices = argsort(sim_scores)[::-1][1:51]
     Get the top 50 most similar movies (skip index 0, which is the movie itself).

  4. Combine with quality: score = alpha * similarity + (1 - alpha) * wr_norm
     Blend content similarity with movie quality (weighted rating).

  5. MMR re-ranking: pick diverse results (not 5 Batman movies in a row).

  6. Return details:  self.smd[['title', 'genres', ...]].iloc[selected_indices]
     Use the stored DataFrame to return full movie information.
```

---

## 6. Hot-Swapping and Runtime Operations

### Hot-Swap: Replacing the Model Without Restarting

The application supports **hot-swapping** — replacing the live recommender model
in memory without restarting the server. This is handled by `_hot_swap_model()`
in `app/main.py`:

```python
def _hot_swap_model(new_rec):
    global recommender, movies_df, offline_cache
    # Preserve poster URLs from current data
    if movies_df is not None and "poster_url" in movies_df.columns:
        poster_map = movies_df.drop_duplicates(subset="id").set_index("id")["poster_url"]
        new_rec.smd["poster_url"] = new_rec.smd["id"].map(poster_map).fillna("")
    movies_df = new_rec.smd.copy()
    recommender = new_rec
    offline_cache = None   # invalidate cached evaluation results
```

The function:

1. Carries forward poster URLs from the currently loaded data (so they are not
   lost when swapping in a freshly trained model that does not have them)
2. Falls back to `poster_path` for movies still missing `poster_url`
3. Replaces the global `recommender` and `movies_df` references atomically
4. Invalidates the offline evaluation cache (since the model changed)

### Background Operations via API

The web app exposes several heavy operations as **background tasks** that run in
daemon threads. Each operation uses a task management system (`_tasks` dict with
thread-safe locking) that tracks progress and supports cancellation.

| Endpoint                             | What it does                                                             |
| ------------------------------------ | ------------------------------------------------------------------------ |
| `POST /api/operations/retrain`       | Full pipeline: make_dataset + build_features + fit + save pkl + hot-swap |
| `POST /api/operations/evaluate`      | Run all evaluation metrics on 100 sampled test movies                    |
| `POST /api/operations/gridsearch`    | Grid search over 256 weight combinations (cancellable)                   |
| `POST /api/operations/fetch-posters` | Scrape TMDB poster URLs (cancellable, rate-limited)                      |
| `POST /api/operations/apply-weights` | Refit model with custom weights + save + hot-swap                        |
| `POST /api/operations/reload-model`  | Load `recommender.pkl` from disk + hot-swap (synchronous)                |
| `POST /api/operations/reload-data`   | Reload `movies_processed.csv` and merge poster URLs (sync)               |

Background tasks expose their state via:

- `GET /api/tasks` — list all tasks with status
- `GET /api/tasks/{id}` — get progress of a specific task
- `POST /api/tasks/{id}/cancel` — request cancellation of a running task

The retrain workflow:

```
POST /api/operations/retrain
        |
        v
  Background thread:
    1. make_dataset()          → load + clean raw CSVs
    2. build_features()        → create text columns
    3. build_recommender()     → fit TF-IDF + cosine sim
    4. pickle.dump()           → save to models/recommender.pkl
    5. _hot_swap_model()       → replace live model
        |
        v
  New model is live (session state preserved)
```

### Model Introspection

The `get_model_stats()` method on `ContentBasedRecommender` returns training
metadata for analytics:

```python
stats = recommender.get_model_stats()
# Returns: total_movies, field_weights, weighted_rating_params,
#          vectorizer_stats (type, vocab_size, weight per field),
#          feature_matrix_shape, feature_matrix_sparsity,
#          cosine_sim_stats (mean, median, std, p95),
#          data_quality (empty_count, fill_rate per field)
```

This is exposed via `GET /api/analytics/model` and powers the dashboard's model
stats panel.

---

## 7. Limitations of This Approach

### Cold start for new movies

If a new movie is released and you add it to the dataset, you must **retrain the
entire model**. There is no way to incrementally add a single movie because:

- The cosine similarity matrix is N x N. Adding one movie means adding a new row and
  a new column, which requires recomputing similarities against all existing movies.
- The vectorizers' vocabularies are fixed. A new movie might introduce new tokens.

To add new movies: update the raw data, re-run `uv run python -m src.models.train_model`,
and restart the app.

### No online learning from user feedback

The model is completely **static**. When a user swipes right on "Inception" and left
on "The Notebook", that preference information is used only for the current session's
movie selection logic. It is **never fed back into the model** to improve future
recommendations for other users.

A production system would incorporate collaborative filtering (learning from many
users' behavior patterns) or retrain periodically with user engagement signals.

### Memory usage: full N x N similarity matrix in RAM

The cosine similarity matrix is dense and large:

```
  9,000 movies:    9,000 x 9,000 x 8 bytes  =  ~617 MB
  20,000 movies:  20,000 x 20,000 x 8 bytes  = ~3.05 GB
  50,000 movies:  50,000 x 50,000 x 8 bytes  = ~19.1 GB
```

This scales quadratically (O(N^2)), which means doubling the number of movies
quadruples the memory requirement. For a larger catalog, you would need approximate
nearest neighbor techniques (like FAISS or Annoy) that avoid storing the full matrix.

### Single-threaded fit

The `cosine_similarity()` call in scikit-learn uses BLAS under the hood, which can
use multiple CPU cores for matrix multiplication. However, the per-field vectorization
loop is single-threaded -- each field is processed sequentially. For this dataset size
(~9,000 movies), this is not a bottleneck, but it would matter at scale.

### Duplicate title blindness

Because the `indices` Series deduplicates by keeping only the first movie with a given
title, any subsequent movie with an identical title becomes unreachable via the
`recommend(title)` interface. The second "Crash" (2004) is invisible if "Crash" (1996)
was indexed first.

### No personalization beyond current session

The model computes the same similarity scores for all users. Personalization happens
only at the application layer (which liked movies to aggregate recommendations from),
not in the model itself.

---

## Summary

```
  TRAINING:     Not gradient descent. Building vocabularies + computing
                pairwise cosine similarities from movie metadata text.

  PERSISTING:   pickle.dump() saves the entire Python object (DataFrame,
                similarity matrix, vectorizers, indices) to a single .pkl file.

  LOADING:      pickle.load() restores the object. Post-processing merges
                poster URLs and coerces data types. Fallback: fit from scratch.

  CORE IDEA:    Movies are vectors of text features. Similar vectors = similar
                movies. The N x N similarity matrix is precomputed so that
                recommendations are instant O(N) lookups at serving time.
```
