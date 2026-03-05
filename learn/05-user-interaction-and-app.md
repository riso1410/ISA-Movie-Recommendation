# 05 - User Interaction and the Web Application

## How MovieMatch Connects the Recommendation Model to Real Users

This document explains how the MovieMatch web application works -- from the
moment a user opens the page, through every swipe, all the way to personalized
recommendations appearing on screen. We will trace the full journey of data
between the browser (frontend) and the server (backend), and explain how the
system adapts to each user's taste without ever retraining the model.

No prior knowledge of web development or machine learning is assumed.

---

## Table of Contents

1. [The Tinder-Style UX Concept](#1-the-tinder-style-ux-concept)
2. [Session State Management](#2-session-state-management)
3. [Movie Selection Strategy -- Three Phases](#3-movie-selection-strategy----three-phases)
4. [The User Profile Recommendation Engine](#4-the-user-profile-recommendation-engine)
5. [Recommendation List Feedback](#5-recommendation-list-feedback)
6. [API Endpoints Explained](#6-api-endpoints-explained)
7. [How the Model "Adapts" to User Preferences](#7-how-the-model-adapts-to-user-preferences)
8. [Limitations and Possible Improvements](#8-limitations-and-possible-improvements)

---

## 1. The Tinder-Style UX Concept

### What the User Sees

When a user opens MovieMatch, they see a single movie card in the center of the
screen. The card displays:

- The movie poster (a large image)
- The title
- Genre badges (e.g., "Action", "Drama")
- A star rating and vote count
- A short plot overview

To the right of the card, there is a recommendations panel that starts empty.

```
+--------------------------------------------------+
|  MovieMatch                  Liked: 0  Disliked: 0|
+--------------------------------------------------+
|                      |                            |
|   +--------------+   |   Your Recommendations     |
|   |              |   |                            |
|   |  [POSTER]    |   |   Like 3 more movies to    |
|   |              |   |   get personalized          |
|   |              |   |   recommendations           |
|   |  The Dark    |   |                            |
|   |  Knight      |   |                            |
|   |  Action|Drama|   |                            |
|   |  * 8.2       |   |                            |
|   +--------------+   |                            |
|                      |                            |
|    [X]        [<3]   |                            |
|                      |                            |
+--------------------------------------------------+
```

### How Swiping Works

The user makes a simple binary decision for each movie:

- **Swipe right** (or click the heart button, or press the right arrow key) =
  **"I like this"**
- **Swipe left** (or click the X button, or press the left arrow key) =
  **"Not for me"**

The card animates off screen in the chosen direction (sliding right with a
rotation for likes, sliding left for dislikes), and a new card slides in.

The drag interaction works like a real card: the user can click and drag the
card left or right. As they drag, a "LIKE" or "NOPE" overlay fades in on the
card, and the card shadow changes color (green for like, red for dislike). If
they drag more than 100 pixels and release, the swipe is registered. If they
drag less than that and release, the card snaps back -- no decision is recorded.

### Why This Interface Design Works

This design borrows from Tinder (the dating app) because it solves several
problems that plague traditional recommendation interfaces:

1. **Low cognitive load.** The user only ever looks at one movie at a time.
   There is no overwhelming grid of 50 options to compare. One card, one
   decision.

2. **Binary feedback is easy.** The user does not need to assign a rating on a
   1-5 scale or write a review. They just answer: "Would I watch this?" Yes or
   no.

3. **Fast feedback loop.** Each decision takes about 1-2 seconds. Within 30
   seconds, the system has enough data (3 likes) to start personalizing.

4. **Engagement through motion.** The swipe animation and card physics create a
   satisfying, almost game-like interaction. Users tend to keep swiping.

5. **Progressive revelation.** The recommendations panel on the right starts
   empty and gradually fills as the user provides more likes. This gives users
   a sense of the system learning about them.

---

## 2. Session State Management

### The Session Dictionary

When the server starts, it creates a single global dictionary to track the
current user's activity:

```python
session: dict = {"liked": [], "disliked": [], "seen": set(), "swipe_log": []}
```

This dictionary has four keys:

| Key         | Type   | Purpose                                            |
| ----------- | ------ | -------------------------------------------------- |
| `liked`     | `list` | Movie IDs the user swiped right on                 |
| `disliked`  | `list` | Movie IDs the user swiped left on                  |
| `seen`      | `set`  | All movie IDs shown to the user (liked + disliked) |
| `swipe_log` | `list` | Detailed log of every interaction, with metadata   |

### The Swipe Log

The `swipe_log` is the richest data structure in the session. Each entry is a
dictionary recording everything about a single user interaction:

```python
{
    "movie_id": 155,
    "direction": "right",           # "right" (like) or "left" (dislike)
    "source": "model",              # "random", "model", or "rec_list"
    "similarity_score": 0.7234,     # cosine sim (None for random picks)
    "combined_score": 0.6891,       # blended score (None for random)
    "popularity": 112.3,            # movie's TMDB popularity
    "genres": ["Action", "Drama"],  # for analytics
    "timestamp": 1709384521.3,      # Unix timestamp
    "swipe_index": 7,               # sequential position in session
}
```

The `source` field is critical — it tells the profile vector engine how much
to weight this interaction (see doc 03, Section 6.3 for the weight table).
Three sources exist:

- **"random"**: Movie was shown during cold start or via exploration (15%)
- **"model"**: Movie was selected by the profile-based recommendation engine
- **"rec_list"**: User clicked like/dislike directly on the sidebar rec list

### Why `liked` and `disliked` Are Lists

Lists preserve the order of actions. The first movie the user liked is at index
0, the second at index 1, and so on. This ordering could be useful for future
features (e.g., giving more weight to recently liked movies), and it also allows
duplicates -- though the current code does not produce them.

### Why `seen` Is a Set

Every time the system picks a new movie to show, it must check: "Have we already
shown this movie?" This check happens many times per request -- once for every
candidate movie. Using a set makes this check extremely fast.

```
Checking "Is movie #12345 in this collection?"

  List (linear search):   O(n) -- must scan every element
  Set  (hash lookup):     O(1) -- instant, regardless of collection size

If the user has seen 500 movies:
  List: up to 500 comparisons per check
  Set:  ~1 comparison per check
```

The `seen` set is the union of `liked` and `disliked`. Every movie the user
interacts with gets added to `seen`, ensuring it never appears again.

### The Global Session Limitation

There is only one `session` dictionary for the entire server. This means:

- If two people open the app at the same time, they share the same session.
  Person A's likes affect Person B's recommendations.
- There is no login, no cookies, no user IDs.
- If the server restarts, all session data is lost (it only lives in memory).

This is a deliberate simplification. For a course project, a single-user
in-memory session avoids the complexity of databases, authentication, and
session management.

### What Happens on Reset

When a user clicks the "Reset" button, the frontend calls `/api/reset`, and the
server clears everything:

```python
session["liked"] = []
session["disliked"] = []
session["seen"] = set()
session["swipe_log"] = []
```

All four collections are replaced with fresh, empty ones. The user is back to
square one -- the next movie they see will be randomly chosen, and the
recommendations panel goes blank. The model itself is unaffected; it stays in
memory exactly as it was.

---

## 3. Movie Selection Strategy -- Three Phases

The most interesting part of the application is how it decides **which movie to
show next**. This happens in the `get_next_movie()` function, and it operates in
three distinct phases.

```
                    How the app picks the next movie
                    ================================

                        User requests next movie
                                  |
                                  v
                     +------------------------+
                     | How many liked movies? |
                     +------------------------+
                          |              |
                       < 3            >= 3
                          |              |
                          v              v
                   +-----------+  +------------------+
                   | PHASE 1   |  | Roll dice:       |
                   | Random    |  | random() < 0.15? |
                   | (weighted |  +------------------+
                   | by        |     |           |
                   | popularity)|   Yes          No
                   +-----------+     |           |
                          |          v           v
                          |   +-----------+ +------------------+
                          |   | PHASE 2   | | PHASE 3          |
                          |   | Explore   | | Profile-based    |
                          |   | (random   | | recommendation   |
                          |   | popular)  | | from user vector |
                          |   +-----------+ +------------------+
                          |          |           |
                          |          |      [empty?]--> fallback to Phase 1
                          v          v           v
                     Return movie card to user
```

### Phase 1: Cold Start (Fewer Than 3 Likes)

When the user first opens the app, the system knows nothing about their taste.
This is called the **cold start problem** -- a fundamental challenge in
recommendation systems. How do you recommend movies to someone you know nothing
about?

MovieMatch's solution: show random movies, but weighted by popularity.

```python
# From get_next_movie():
unseen = movies_df[~movies_df["id"].isin(seen)]
weights = unseen["popularity"].clip(lower=0.1)
weights = weights / weights.sum()
chosen = unseen.sample(n=1, weights=weights)
```

Let us break this down step by step:

**Step 1: Filter out seen movies.**
`movies_df[~movies_df["id"].isin(seen)]` returns only movies the user has not
yet interacted with.

**Step 2: Create popularity weights.**
Each movie in the dataset has a `popularity` score (from TMDB -- The Movie
Database). Higher values mean more people know about the movie. For example:

| Movie                   | Popularity |
| ----------------------- | ---------- |
| The Dark Knight         | 112.3      |
| Inception               | 95.7       |
| Some obscure indie film | 0.3        |
| A movie with bad data   | 0.0        |

**Step 3: Clip weights to a minimum of 0.1.**
`popularity.clip(lower=0.1)` ensures no movie has a weight of zero or negative.
Why? Because `np.random.choice` (used internally by `.sample()`) cannot handle
zero-probability items -- they would never be selected. Even obscure movies
deserve a small chance of appearing.

After clipping:

| Movie                   | Weight |
| ----------------------- | ------ |
| The Dark Knight         | 112.3  |
| Inception               | 95.7   |
| Some obscure indie film | 0.3    |
| A movie with bad data   | 0.1    |

**Step 4: Normalize weights to sum to 1.**
`weights = weights / weights.sum()` converts raw weights into probabilities.
If the sum of all weights is 208.4:

| Movie                   | Probability |
| ----------------------- | ----------- |
| The Dark Knight         | 53.9%       |
| Inception               | 45.9%       |
| Some obscure indie film | 0.14%       |
| A movie with bad data   | 0.05%       |

(This is a simplified example with only 4 movies. In reality, probabilities are
spread across thousands of movies.)

**Step 5: Sample one movie.**
`unseen.sample(n=1, weights=weights)` randomly picks one movie, with popular
movies being much more likely to be chosen.

**Why popularity weighting?**

The goal during cold start is to show movies the user is likely to _recognize_.
If the first movie shown is an obscure 1987 Bulgarian art film, most users will
just dislike it -- not because they would not enjoy it, but because they have
never heard of it and cannot judge it from a single card. Popular movies like
"The Dark Knight" or "Inception" are ones most users have seen or at least know
about, so they can make a genuine like/dislike decision.

This gives the recommendation engine useful signal to work with.

### Phase 2: Exploration (15% of Model Phase)

Once the user has 3+ likes, 85% of the time the model picks the next movie.
But **15% of the time**, the system intentionally shows a random
popularity-weighted movie instead — even though the model has a recommendation
ready.

```python
if random.random() < EXPLORATION_RATE:  # EXPLORATION_RATE = 0.15
    # Show a random movie (same as cold start)
    ...
    return movie_to_dict(chosen.iloc[0], source="random")
```

This is called **epsilon-greedy exploration**, borrowed from reinforcement
learning (see doc 03, Section 8 for the full explanation). The random pick
prevents the system from trapping the user in an echo chamber where they only
see the same type of movie repeatedly.

These exploration movies are tagged with `source="random"` in the swipe log, so
the profile vector engine weights them appropriately (lower weight than model
picks).

### Phase 3: Profile-Based Recommendations (85% of Model Phase)

The remaining 85% of the time, the system uses the **user profile vector** to
find the best next movie. This replaces the old per-title aggregation approach.

```python
sorted_recs = _profile_recommendations(n=30)
for mid, combined_score, cosine_score in sorted_recs:
    movie_row = movies_df[movies_df["id"] == mid]
    if not movie_row.empty:
        return movie_to_dict(movie_row.iloc[0], source="model")
```

The `_profile_recommendations()` function:

1. Builds a weighted sum of all interacted movies' feature vectors
2. L2-normalizes the result
3. Computes cosine similarity against every movie in the catalog
4. Blends similarity with quality (combined score)
5. Excludes seen movies, penalizes disliked-similar movies
6. MMR re-ranks for diversity

This is documented in detail in doc 03, Sections 6-7.

**Why 3 likes?** With just 1 or 2 liked movies, the profile vector is too
sparse. If a user likes only "The Dark Knight", the profile is essentially a
copy of that single movie's feature vector, so every recommendation will be a
Batman or Christopher Nolan film. With 3 likes, there is enough variety to
triangulate the user's taste.

**Fallback to random.** If the profile engine returns no unseen
recommendations (extremely rare — would require the user to have seen nearly
every movie), the system falls back to Phase 1's popularity-weighted random
selection.

---

## 4. The User Profile Recommendation Engine

This is the core algorithm that makes MovieMatch feel "smart". Instead of
querying the model once per liked movie and merging results, the system builds
a **single user profile vector** that represents the user's overall taste.

The mathematical details are covered in doc 03, Sections 6-8. This section
focuses on _how and when_ the profile engine is used in the application.

### The Concept: User as a Movie

Each movie in the catalog is a sparse feature vector (around 20,000 dimensions —
from TF-IDF overview words, genre labels, cast names, etc.). The user profile is
built by taking a **weighted sum** of the feature vectors of every movie the user
has interacted with:

```
profile = Σ  feature_vector[movie_i] × feedback_weight_i
         i∈swipe_log
```

The result is a vector in the same space as the movies. We can then compute
cosine similarity between this "virtual movie" (the user's taste) and every
real movie to rank the entire catalog.

### Feedback Weights: Not All Interactions Are Equal

The weight assigned to each interaction depends on its **direction** (like vs
dislike) and **source** (how the user encountered the movie):

```
Action                          Weight    Intuition
--------------------------------------------------------------
Like a random movie              +1.0    Baseline positive signal
Dislike a random movie           -0.3    Weak negative (ambiguous)
Like a model recommendation      +2.5    Strong — model was right
Dislike a model recommendation   -1.2    Model was wrong, learn from it
Like from recommendation list    +3.0    Strongest — most intentional
Dislike from recommendation list -1.5    Strong negative signal
```

Key design decisions:

- **Asymmetry**: Negative weights have smaller magnitude. Dislikes are noisier
  (user may not recognize the movie, or simply not be in the mood).
- **Source hierarchy**: Random < Model < Rec List. The more intentional the
  action, the more we trust the signal.

### A Worked Example

Suppose the user has swiped on 5 movies:

```
1. "The Dark Knight"    → right, random     weight = +1.0
2. "Scary Movie"        → left,  random     weight = -0.3
3. "Inception"          → right, model      weight = +2.5
4. "Hostel"             → left,  model      weight = -1.2
5. "The Prestige"       → right, rec_list   weight = +3.0
```

The profile vector becomes:

```
profile = TDK_vec × 1.0
        + Scary_vec × (-0.3)
        + Inception_vec × 2.5
        + Hostel_vec × (-1.2)
        + Prestige_vec × 3.0
```

The resulting vector is strong in "Nolan", "thriller", "crime", "twist" (shared
by the three liked movies with high weights). Horror dimensions are subtracted
by both "Scary Movie" and "Hostel" negatives.

When we compute cosine similarity between this profile and every movie:

- "Memento" scores high (Nolan, thriller, twist — matches the profile)
- "The Texas Chain Saw Massacre" scores low (horror — opposite of profile)
- "Interstellar" scores moderately (Nolan, drama — partial match)

### After Scoring: Post-Processing

After computing `profile × feature_matrix` cosine scores:

1. **Combined score** blends similarity with quality:
   `0.7 × cosine + 0.3 × wr_norm`

2. **Hard exclude**: Seen movies are set to `-inf` (never shown again)

3. **Soft penalty**: Movies with cosine_sim > 0.7 to any disliked movie get
   their score halved (see doc 03, Section 7)

4. **MMR re-ranking**: Top candidates are diversity-filtered to avoid a list
   of near-identical movies (see doc 03, Section 4)

### Why This Is Better Than Per-Title Aggregation

```
Old: Per-Title Aggregation           New: Profile Vector
─────────────────────────           ───────────────────
5 liked movies                       5 likes + 3 dislikes
  → 5 × recommend()                   → 1 weighted vector sum
  → Only likes used                   → Both likes AND dislikes
  → All likes equal weight            → Source-aware weighting
  → Results merged by score sum       → Cosine sim vs entire catalog
  → No dislike penalty                → Soft penalty for disliked-similar
  → No exploration                    → 15% epsilon-greedy exploration
```

---

## 5. Recommendation List Feedback

### The Sidebar as an Interactive Element

The recommendation panel on the right side of the screen is not just a passive
display — users can **like or dislike** movies directly from it. Hovering over
a recommendation card reveals two overlay buttons:

```
+-------------------+
|  [POSTER]         |
|                   |
|  +---+     +---+  |   ← Appears on hover
|  | ✘ |     | ♥ |  |
|  +---+     +---+  |
|                   |
|  "Inception"      |
|  Action | Sci-Fi  |
|  ★ 8.1            |
+-------------------+
```

Clicking ✘ (dislike) or ♥ (like) triggers a flash animation (green for like,
red for dislike), removes the card, and immediately refreshes the rec list.

### Why Rec List Feedback Matters

Rec list feedback carries the **highest weight** (3.0 for likes, -1.5 for
dislikes) because it is the most intentional form of interaction:

1. The user is looking at a curated recommendation — not a random movie
2. They make an explicit evaluation without being "forced" to swipe
3. Liking from the rec list confirms the model is working well
4. Disliking from the rec list is a strong signal the model got it wrong

### The /api/rec-feedback Endpoint

When the user clicks a button on a rec card:

1. `POST /api/rec-feedback` is called with `{movie_id, direction}`
2. The movie is added to `swipe_log` with `source: "rec_list"`
3. State is updated: `liked`/`disliked`/`seen`
4. `_profile_recommendations()` is called again to rebuild the rec list
5. The refreshed list is returned inline (no extra API call needed)

A **duplicate guard** prevents stale UI from corrupting state: if the movie is
already in `seen`, the server skips state mutation and just returns fresh recs.

### How Rec Feedback Affects Recommendations

Because rec-list likes carry weight 3.0 (the highest), a single rec-list like
has more impact on the profile than three random swipes:

```
1 rec-list like (weight 3.0)  >  3 random likes (3 × 1.0 = 3.0)
```

Similarly, a rec-list dislike (-1.5) strongly pushes the profile away from that
type of movie and triggers the soft penalty for similar movies.

---

## 6. API Endpoints Explained

The frontend and backend communicate through HTTP endpoints (API routes). The
core user-facing endpoints handle swiping, movie selection, and recommendations.
Additional analytics, operations, and task management endpoints power the
dashboard (see Section 6.2).

### Overview: Request Flow

```
  Browser (React Frontend)            Server (FastAPI Backend)
  ========================            =======================

  User swipes right on a movie
          |
          |  POST /api/swipe
          |  {"movie_id": 155, "direction": "right"}
          +------------------------------------------>
          |                                          Log to swipe_log
          |                                          Update liked/seen
          <------------------------------------------+
          |  {"status": "ok"}
          |
          |  GET /api/movie
          +------------------------------------------>
          |                                          >= 3 likes?
          |                                          → 15% exploration
          |                                          → 85% profile vector
          |                                          < 3 likes?
          |                                          → Random weighted
          <------------------------------------------+
          |  {"id": 49026, "title": "...", "source": "model"}
          |
          |  GET /api/recommendations
          +------------------------------------------>
          |                                          Build profile vector
          |                                          Score entire catalog
          |                                          MMR re-rank top 20
          <------------------------------------------+
          |  {"recommendations": [...], "liked_count": 4}
          |
  User clicks ♥ on a rec card
          |
          |  POST /api/rec-feedback
          |  {"movie_id": 550, "direction": "right"}
          +------------------------------------------>
          |                                          Log with source=rec_list
          |                                          Update liked/seen
          |                                          Rebuild recs
          <------------------------------------------+
          |  {"status": "ok", "recommendations": [...]}
```

### POST /api/swipe

**Purpose:** Record a user's like or dislike on the main movie card.

**Request:**

```json
{
  "movie_id": 155,
  "direction": "right"
}
```

**Server logic:**

1. Build a detailed `swipe_log` entry with source, scores, genres, timestamp
2. Append to `session["swipe_log"]`
3. Append movie ID to `session["liked"]` or `session["disliked"]`
4. Add to `session["seen"]`

The log entry captures the `source` from `pending_movie_meta` (set when the
movie was selected in `get_next_movie()`), so the profile engine knows whether
this was a random, model, or exploration pick.

**Response:**

```json
{ "status": "ok" }
```

### POST /api/rec-feedback

**Purpose:** Record feedback on a recommendation list card.

**Request:**

```json
{
  "movie_id": 550,
  "direction": "left"
}
```

**Server logic:**

1. **Duplicate guard**: If movie is already in `seen`, skip state mutation
   (prevents stale UI from corrupting state)
2. Build log entry with `source: "rec_list"`
3. Update `swipe_log`, `liked`/`disliked`, `seen`
4. Call `_profile_recommendations(n=20)` to get refreshed rec list
5. Return the new rec list inline — no extra API call needed

**Response:**

```json
{
  "status": "ok",
  "recommendations": [
    {
      "id": 155,
      "title": "The Dark Knight",
      "similarity_score": 0.7234,
      "combined_score": 0.6891,
      ...
    }
  ],
  "liked_count": 5
}
```

### GET /api/movie

**Purpose:** Return the next movie card for the user to swipe on.

**Request:** No body or parameters. Just a GET request.

**Server logic:** The three-phase selection strategy from Section 3:
cold start → exploration (15%) → profile-based recommendation (85%).
Each response includes a `source` field so the frontend can display
an "Exploring" or "Model" badge.

**Response (success):**

```json
{
  "id": 49026,
  "title": "The Dark Knight Rises",
  "overview": "Following the death of District Attorney Harvey Dent...",
  "genres": ["Action", "Crime", "Drama", "Thriller"],
  "vote_average": 7.6,
  "vote_count": 9263,
  "poster_url": "https://image.tmdb.org/t/p/w500/...",
  "source": "model"
}
```

**Response (no movies left):**

```json
{ "error": "No more movies to show!" }
```

### GET /api/recommendations

**Purpose:** Return the top 20 recommendations for the sidebar panel.

**Request:** No parameters.

**Server logic:** Calls `_profile_recommendations(n=20)` which builds the user
profile vector, scores the entire catalog, and returns the top 20 after MMR
re-ranking. Each result includes `similarity_score` (cosine sim between the
user's profile and the movie) and `combined_score` (blended with quality).

**Response:**

```json
{
  "recommendations": [
    {
      "id": 155,
      "title": "The Dark Knight",
      "overview": "Batman raises the stakes...",
      "genres": ["Drama", "Action", "Crime", "Thriller"],
      "vote_average": 8.2,
      "vote_count": 12269,
      "poster_url": "...",
      "similarity_score": 0.7234,
      "combined_score": 0.6891
    }
  ],
  "liked_count": 4
}
```

### GET /api/stats

**Purpose:** Return session statistics for the header display.

**Response:**

```json
{
  "liked": 4,
  "disliked": 7,
  "recommendations_count": 20
}
```

### GET /api/reset

**Purpose:** Clear all session data and start fresh.

**Server logic:** Resets `liked`, `disliked`, `seen`, and `swipe_log` to empty.

**Response:**

```json
{ "status": "reset" }
```

### 6.2 Analytics, Operations, and Dashboard Endpoints

Beyond the core swipe/recommend flow, the app exposes endpoints for analytics,
model operations, and background task management. These power the dashboard UI
at `/dashboard`.

#### Analytics Endpoints

| Endpoint                     | Purpose                                                                                                                                                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /api/analytics`         | Session analytics: like rates by source, confusion matrix (TP/FP/FN/TN), swipe buckets, genre distribution, popularity stats, score distributions, calibration curves |
| `GET /api/analytics/catalog` | Catalog-level: genre distribution, popularity histogram                                                                                                               |
| `GET /api/analytics/model`   | Model stats: field weights, vocab sizes, sparsity, cosine sim distribution histogram                                                                                  |
| `GET /api/analytics/offline` | Offline evaluation metrics (cached after first call): Precision@K, NDCG@K, MRR, Serendipity, Coverage, ILD, Novelty, per-genre precision                              |

The session analytics endpoint computes:

- **Like rates** by source (random, model, rec_list) and model lift (model rate / random rate)
- **Confusion matrix**: TP = liked model/rec_list picks, FP = disliked model/rec_list picks, FN = liked random, TN = disliked random. Derives precision, recall, F1.
- **Swipe buckets**: Like rate in groups of 5 swipes, showing how engagement evolves over time
- **Calibration**: Similarity score bins with observed like rate (does higher score = higher like probability?)

#### Operations Endpoints

These trigger heavy operations, either as background tasks (retrain, evaluate,
gridsearch, fetch-posters, apply-weights) or synchronous quick actions
(reload-model, reload-data). See doc 04, Section 6 for details.

| Endpoint                             | Type       | What it does                             |
| ------------------------------------ | ---------- | ---------------------------------------- |
| `POST /api/operations/retrain`       | Background | Full pipeline + hot-swap model           |
| `POST /api/operations/evaluate`      | Background | Run all evaluation metrics               |
| `POST /api/operations/gridsearch`    | Background | 256-combo grid search (cancellable)      |
| `POST /api/operations/fetch-posters` | Background | Scrape TMDB posters (cancellable)        |
| `POST /api/operations/apply-weights` | Background | Refit with custom weights + hot-swap     |
| `POST /api/operations/reload-model`  | Sync       | Load pkl from disk                       |
| `POST /api/operations/reload-data`   | Sync       | Reload processed CSV + merge poster URLs |
| `GET /api/model/weights`             | Sync       | Return current field weights             |

#### Task Management Endpoints

Background operations create tasks with unique IDs. The dashboard polls these
for progress updates.

| Endpoint                      | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| `GET /api/tasks`              | List all tasks with status + active task |
| `GET /api/tasks/{id}`         | Get progress of a specific task          |
| `POST /api/tasks/{id}/cancel` | Request cancellation of a running task   |

#### The Dashboard

The dashboard (`/dashboard`, served from `app/static/dashboard.html`) provides
a visual interface for model analytics, session analytics, and operations. It
is a separate single-file HTML page (like the main swipe UI) that communicates
with the same FastAPI backend.

---

## 7. How the Model "Adapts" to User Preferences

This is a critical distinction that is easy to misunderstand.

### What Does NOT Happen

The recommendation model (the TF-IDF vectors, the cosine similarity matrix) is
**never retrained** during a user session. The model was trained once (during
`train_model.py` or at server startup) and sits in memory as a fixed
mathematical structure. When a user swipes, no neural network is being updated,
no weights are being adjusted, no new patterns are being learned.

### What DOES Happen

The "adaptation" happens through the **user profile vector**. Each interaction
modifies the profile, which changes how the fixed model's feature space is
queried:

```
Session starts:
  Model: [fixed feature matrix + cosine sim for ~4600 movies]
  Profile: [zero vector]
  --> No signal. Show random movies.

User likes "The Dark Knight" (random, weight +1.0):
  Model: [unchanged]
  Profile: [1.0 × TDK_features]
  --> Profile = "TDK-like". But < 3 likes, still random.

User dislikes "Scary Movie" (random, weight -0.3):
  Model: [unchanged]
  Profile: [1.0 × TDK - 0.3 × Scary]
  --> Profile shifts AWAY from horror/comedy. Still random.

User likes "Inception" (model, weight +2.5):
  Model: [unchanged]
  Profile: [1.0 × TDK - 0.3 × Scary + 2.5 × Inception]
  --> 3+ likes! Profile is now "Nolan thrillers, not horror".
  --> Scores entire catalog via cosine(profile, all movies).

User likes "Memento" from rec list (weight +3.0):
  Model: [unchanged]
  Profile: [... + 3.0 × Memento]
  --> Profile strongly reinforces "twist endings, cerebral".
  --> Soft penalty active against horror-similar movies.
```

### The Key Insight

The model is a **fixed map** of movie relationships. The profile vector is a
**moving pointer** on that map. Each swipe adjusts where the pointer aims.
Likes push it toward clusters of similar movies; dislikes push it away.

This is different from the old per-title approach where each like was a
separate query:

```
Old: 5 liked movies → 5 separate queries → merge results
New: 5 likes + 3 dislikes → 1 weighted profile → 1 query over entire catalog
```

### Session-Based, Non-Learning

In recommendation system terminology, this approach is:

- **Session-based:** Preferences only exist for the current session. Close the
  browser, lose your profile.
- **Non-learning:** The model itself does not update. It is a static feature
  matrix and similarity lookup.
- **Profile-based:** A single user profile vector summarizes all feedback and
  is compared against the entire catalog.
- **Feedback-weighted:** Different actions carry different evidence strength.

This is different from systems like Netflix or Spotify, which:

- Store your entire history permanently
- Periodically retrain models on millions of users' data
- Use collaborative filtering ("users like you also watched...")

---

## 8. Limitations and Possible Improvements

### Current Limitations

**1. No persistence.** All session data lives in a Python dictionary in RAM.
If the server restarts (crash, deployment, reboot), every user's history
vanishes. There is no database.

**2. Single global session.** There is one `session` dict for the entire
application. If two users open the app simultaneously, they share likes and
dislikes. User A's taste corrupts User B's recommendations. There are no user
accounts, no cookies, no authentication.

**3. No collaborative filtering.** The system only uses content similarity
("this movie has similar genres, cast, and keywords to what you liked"). It
never considers what other users with similar tastes have enjoyed. In practice,
collaborative filtering often produces better "surprise" recommendations --
movies you would not have found through content similarity alone.

**4. Cold start quality.** During Phase 1, popularity weighting helps, but
the user might still see movies they find irrelevant. There is no genre
preference survey, no "pick 5 movies you like" onboarding flow.

**5. Fixed feedback weights.** The weights (1.0, 2.5, 3.0, etc.) were
hand-tuned based on intuition. In a production system, these would be learned
from A/B tests or optimized against engagement metrics.

### What Has Been Addressed

Several limitations from earlier versions of the system have been resolved:

| Former Limitation                  | Current Solution                                            |
| ---------------------------------- | ----------------------------------------------------------- |
| Dislikes were wasted               | Profile vector uses dislikes as negative weight             |
| No penalty for similar-to-disliked | Soft penalty halves scores for movies sim > 0.7 to disliked |
| Echo chamber risk                  | 15% epsilon-greedy exploration breaks filter bubbles        |
| All likes treated equally          | Source-aware weights: random < model < rec_list             |
| Rec list was view-only             | Interactive like/dislike buttons on rec cards               |

### Possible Improvements

| Improvement                           | Difficulty | Impact |
| ------------------------------------- | ---------- | ------ |
| Per-user sessions (cookies)           | Low        | High   |
| Database persistence (SQLite)         | Low        | High   |
| Onboarding genre picker               | Low        | Medium |
| Collaborative filtering               | High       | High   |
| Temporal decay (recent > older likes) | Medium     | Medium |
| Implicit signals (hover time, scroll) | Medium     | Low    |
| A/B testing of algorithms             | High       | Medium |
| Learned feedback weights              | Medium     | Medium |

---

## Summary

The MovieMatch application takes a pre-trained content-based recommendation
model and wraps it in an interactive feedback loop:

```
  +----------+     swipe/     +-----------+     profile    +-------+
  |  User    |  rec-feedback  |  Session  |     vector     | Model |
  |  (browser|  ----------->  |  State    |  ----------->  | (fixed|
  |   card + |                |  (log,    |                |  feat |
  |   rec    |  <-----------  |   liked,  |  <-----------  |  mtx) |
  |   list)  |   next movie   |   seen)   |  cosine sim    +-------+
  +----------+   + recs list   +-----------+                    |
                                    |                           |
                                    v                           |
                              +-----------+                     |
                              | Feedback  |     Exploration     |
                              | Weights   |     15% random      |
                              | + Soft    |     breaks echo     |
                              | Penalty   |     chambers        |
                              +-----------+                     |
```

The model never changes. The user profile vector grows and shifts with each
interaction. As more feedback accumulates, the profile becomes a richer
representation of the user's taste — capturing both what they like and what
they dislike. The exploration mechanism ensures the system does not become
trapped in a narrow slice of the catalog.
