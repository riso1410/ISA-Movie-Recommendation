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
3. [Movie Selection Strategy -- The Two Phases](#3-movie-selection-strategy----the-two-phases)
4. [The Recommendation Aggregation Algorithm](#4-the-recommendation-aggregation-algorithm)
5. [API Endpoints Explained](#5-api-endpoints-explained)
6. [How the Model "Adapts" to User Preferences](#6-how-the-model-adapts-to-user-preferences)
7. [Limitations and Possible Improvements](#7-limitations-and-possible-improvements)

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
session: dict = {"liked": [], "disliked": [], "seen": set()}
```

This dictionary has three keys:

| Key        | Type   | Purpose                                            |
| ---------- | ------ | -------------------------------------------------- |
| `liked`    | `list` | Movie IDs the user swiped right on                 |
| `disliked` | `list` | Movie IDs the user swiped left on                  |
| `seen`     | `set`  | All movie IDs shown to the user (liked + disliked) |

### Why `liked` and `disliked` Are Lists

Lists preserve the order of actions. The first movie the user liked is at index
0, the second at index 1, and so on. This ordering could be useful for future
features (e.g., giving more weight to recently liked movies), and it also allows
duplicates -- though the current code does not produce them.

### Why `seen` Is a Set

Every time the system picks a new movie to show, it must check: "Have we already
shown this movie?" This check happens many times per request -- once for every
candidate movie. Using a set makes this check extremely fast.

Here is the performance difference:

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
server does this:

```python
session["liked"] = []
session["disliked"] = []
session["seen"] = set()
```

All three collections are replaced with fresh, empty ones. The user is back to
square one -- the next movie they see will be randomly chosen, and the
recommendations panel goes blank. The model itself is unaffected; it stays in
memory exactly as it was.

---

## 3. Movie Selection Strategy -- The Two Phases

The most interesting part of the application is how it decides **which movie to
show next**. This happens in the `get_next_movie()` function, and it operates in
two distinct phases.

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
                   | PHASE 1   |  | PHASE 2          |
                   | Random    |  | Content-based    |
                   | (weighted |  | recommendations  |
                   | by        |  | aggregated from  |
                   | popularity)|  | all liked movies |
                   +-----------+  +------------------+
                          |              |
                          |         [empty?]----> fallback to Phase 1
                          |              |
                          v              v
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

### Phase 2: Warm Recommendations (3 or More Likes)

Once the user has liked at least 3 movies, the system switches to content-based
recommendations. Instead of random picks, it now uses the trained model to find
movies similar to what the user likes.

```python
if len(liked_titles) >= 3:
    sorted_recs = _aggregate_recommendations(liked_titles, n_per_title=15)
    for mid, _ in sorted_recs:
        movie_row = movies_df[movies_df["id"] == mid]
        if not movie_row.empty:
            return movie_to_dict(movie_row.iloc[0])
```

**Why 3 likes?** With just 1 or 2 liked movies, the recommendation source is
too narrow. If a user likes only "The Dark Knight", every recommendation will be
a Batman or Christopher Nolan film. With 3 likes, there is enough variety to
triangulate the user's taste. For example, liking "The Dark Knight",
"Inception", and "The Shawshank Redemption" reveals an interest in
well-crafted, serious dramas with complex plots -- not just superhero movies.

**Fallback to random.** If the content-based engine runs out of unseen
recommendations (all suggestions have already been shown), the system falls back
to Phase 1's popularity-weighted random selection. This ensures the user always
gets a new movie card, even if the model has exhausted its ideas.

---

## 4. The Recommendation Aggregation Algorithm

This is the core algorithm that makes MovieMatch feel "smart". Let us walk
through it with a concrete example.

### Setup: The User's Likes

Suppose the user has liked three movies:

1. "The Dark Knight" (superhero, crime, drama, Nolan)
2. "Inception" (sci-fi, thriller, Nolan)
3. "Interstellar" (sci-fi, drama, space, Nolan)

### Step 1: Get Recommendations for Each Liked Movie

The system calls `recommender.recommend(title, n=15)` for each liked movie.
Each call returns the top 15 most similar movies (using the cosine similarity
matrix built during training). Each result includes a `similarity_score`
(how content-similar the movie is) and a `weighted_rating` (IMDB-style quality
score).

```
Recs for "The Dark Knight" (top 5 of 15):
  Batman Begins          similarity=0.82  weighted_rating=7.5
  The Dark Knight Rises  similarity=0.78  weighted_rating=7.2
  Memento                similarity=0.41  weighted_rating=7.8
  Heat                   similarity=0.38  weighted_rating=7.6
  Prestige, The          similarity=0.35  weighted_rating=7.9

Recs for "Inception" (top 5 of 15):
  Shutter Island         similarity=0.55  weighted_rating=7.6
  Memento                similarity=0.52  weighted_rating=7.8
  The Prestige           similarity=0.49  weighted_rating=7.9
  The Matrix             similarity=0.43  weighted_rating=7.8
  Interstellar           similarity=0.40  weighted_rating=8.1  <-- already seen

Recs for "Interstellar" (top 5 of 15):
  Gravity                similarity=0.61  weighted_rating=7.1
  The Martian            similarity=0.55  weighted_rating=7.3
  Memento                similarity=0.38  weighted_rating=7.8
  The Prestige           similarity=0.36  weighted_rating=7.9
  2001: A Space Odyssey  similarity=0.34  weighted_rating=7.8
```

### Step 2: Aggregate Scores Across All Sources

Here is the key insight: a movie that appears in multiple recommendation lists
is probably a very good match for this user. The `_aggregate_recommendations()`
function accumulates scores.

For each recommended movie, the score formula is:

```
score += similarity_score + 0.1 * weighted_rating
```

The `0.1 * weighted_rating` term is a small quality bonus. A movie with a
weighted rating of 7.8 gets an extra 0.78 added to its score. This gently
pushes higher-quality movies up the list, without overwhelming the similarity
signal.

Let us trace "Memento" through the aggregation:

```
Source: "The Dark Knight" -->  score += 0.41 + 0.1 * 7.8 = 0.41 + 0.78 = 1.19
Source: "Inception"       -->  score += 0.52 + 0.1 * 7.8 = 0.52 + 0.78 = 1.30
Source: "Interstellar"    -->  score += 0.38 + 0.1 * 7.8 = 0.38 + 0.78 = 1.16
                               -------
                       Total:  3.65
```

Now let us trace "The Prestige":

```
Source: "The Dark Knight" -->  score += 0.35 + 0.1 * 7.9 = 0.35 + 0.79 = 1.14
Source: "Inception"       -->  score += 0.49 + 0.1 * 7.9 = 0.49 + 0.79 = 1.28
Source: "Interstellar"    -->  score += 0.36 + 0.1 * 7.9 = 0.36 + 0.79 = 1.15
                               -------
                       Total:  3.57
```

And "Batman Begins" (only appears from one source):

```
Source: "The Dark Knight" -->  score += 0.82 + 0.1 * 7.5 = 0.82 + 0.75 = 1.57
                               -------
                       Total:  1.57
```

### Step 3: Exclude Seen Movies and Sort

Any movie the user has already swiped on (in the `seen` set) is skipped. Then
the remaining movies are sorted by aggregate score in descending order.

```
Final ranking (simplified):
  1. Memento              score = 3.65  (recommended by 3 liked movies)
  2. The Prestige         score = 3.57  (recommended by 3 liked movies)
  3. Shutter Island       score = 1.31  (recommended by 1 liked movie)
  4. Batman Begins        score = 1.57  (recommended by 1 liked movie)
  5. Gravity              score = 1.32  (recommended by 1 liked movie)
  ...
```

The system returns the #1 ranked movie as the next card.

### Why Multi-Source Aggregation Works

```
Single-source approach:          Multi-source aggregation:

User likes "The Dark Knight"     User likes "The Dark Knight",
                                 "Inception", "Interstellar"
          |                                   |
          v                                   v
   Only Batman/Nolan films       Movies connected to MULTIPLE
   are recommended               liked movies bubble up
                                              |
          |                                   v
          v                       "Memento" rises to #1 because
   Narrow, predictable            it's similar to ALL THREE
   recommendations                liked movies -- not just one
```

The multi-source approach captures the _intersection_ of the user's interests.
A user who likes "The Dark Knight", "Inception", and "Interstellar" is not
just a Nolan fan -- they like complex, cerebral thrillers. "Memento" fits
that pattern perfectly, and the aggregation algorithm surfaces it precisely
because it is connected to all three liked movies.

---

## 5. API Endpoints Explained

The frontend and backend communicate through five HTTP endpoints (API routes).
Here is the complete request/response flow for each one.

### Overview: Request Flow

```
  Browser (React Frontend)            Server (FastAPI Backend)
  ========================            =======================

  User swipes right on a movie
          |
          |  POST /api/swipe
          |  {"movie_id": 155, "direction": "right"}
          +------------------------------------------>
          |                                          Update session
          |                                          liked.append(155)
          |                                          seen.add(155)
          <------------------------------------------+
          |  {"status": "ok"}
          |
          |  GET /api/movie
          +------------------------------------------>
          |                                          Check liked count
          |                                          >= 3? Aggregate recs
          |                                          < 3?  Random weighted
          <------------------------------------------+
          |  {"id": 49026, "title": "The Dark Knight Rises", ...}
          |
          |  GET /api/recommendations
          +------------------------------------------>
          |                                          For each liked movie:
          |                                            get top-10 similar
          |                                          Aggregate, sort, top 20
          <------------------------------------------+
          |  {"recommendations": [...], "liked_count": 4}
          |
          |  GET /api/stats
          +------------------------------------------>
          |                                          Count liked, disliked,
          |                                          available recs
          <------------------------------------------+
          |  {"liked": 4, "disliked": 7, "recommendations_count": 38}
```

### POST /api/swipe

**Purpose:** Record a user's like or dislike on a movie.

**Request:**

```json
{
  "movie_id": 155,
  "direction": "right"
}
```

- `movie_id`: The integer ID of the movie being rated.
- `direction`: Either `"right"` (like) or `"left"` (dislike).

**Server logic:**

```python
if req.direction == "right":
    session["liked"].append(req.movie_id)
else:
    session["disliked"].append(req.movie_id)
session["seen"].add(req.movie_id)
```

The movie ID is always added to `seen`, regardless of direction. It is also
added to either `liked` or `disliked`.

**Response:**

```json
{ "status": "ok" }
```

### GET /api/movie

**Purpose:** Return the next movie card for the user to swipe on.

**Request:** No body or parameters. Just a GET request.

**Server logic:** This is the two-phase selection strategy described in
Section 3. If there are 3+ liked movies, it runs the aggregation algorithm.
Otherwise, it picks a random popularity-weighted movie.

**Response (success):**

```json
{
  "id": 49026,
  "title": "The Dark Knight Rises",
  "overview": "Following the death of District Attorney Harvey Dent...",
  "genres": ["Action", "Crime", "Drama", "Thriller"],
  "vote_average": 7.6,
  "vote_count": 9263,
  "poster_url": "https://image.tmdb.org/t/p/w500/dEYnvnUfXrqvqeRSqvIEtmzhoA8.jpg"
}
```

**Response (no movies left):**

```json
{ "error": "No more movies to show!" }
```

Note: The `overview` field is truncated to 500 characters on the server side
to keep responses compact.

### GET /api/recommendations

**Purpose:** Return the top 20 recommendations for the sidebar panel.

**Request:** No parameters.

**Server logic:** This uses a slightly different aggregation than
`get_next_movie()`. For each liked movie, it fetches the top 10 similar movies
(not 15). It aggregates by `weighted_rating` (not `similarity_score + 0.1 *
weighted_rating`). It also tracks how many liked movies each recommendation was
sourced from (the `count` field). The top 20 by aggregate weighted rating are
returned.

**Response (with recommendations):**

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
      "poster_url": "https://image.tmdb.org/t/p/w500/..."
    }
  ],
  "liked_count": 4
}
```

**Response (no likes yet):**

```json
{
  "recommendations": [],
  "liked_count": 0
}
```

### GET /api/stats

**Purpose:** Return session statistics for the header display.

**Request:** No parameters.

**Server logic:** Counts liked movies, disliked movies, and the total number
of unique recommendations currently available (not yet seen by the user).

**Response:**

```json
{
  "liked": 4,
  "disliked": 7,
  "recommendations_count": 38
}
```

### GET /api/reset

**Purpose:** Clear all session data and start fresh.

**Request:** No parameters.

**Server logic:**

```python
session["liked"] = []
session["disliked"] = []
session["seen"] = set()
```

**Response:**

```json
{ "status": "reset" }
```

After this, the frontend clears its local state (recommendations list and stats
counters) and fetches a new random movie.

---

## 6. How the Model "Adapts" to User Preferences

This is a critical distinction that is easy to misunderstand.

### What Does NOT Happen

The recommendation model (the TF-IDF vectors, the cosine similarity matrix) is
**never retrained** during a user session. The model was trained once (during
`train_model.py` or at server startup) and sits in memory as a fixed
mathematical structure. When a user swipes, no neural network is being updated,
no weights are being adjusted, no new patterns are being learned.

### What DOES Happen

The "adaptation" is entirely about **which queries** are sent to the fixed
model. Think of the model as a library catalog and the user's likes as search
queries:

```
Session starts:
  Model: [fixed cosine similarity matrix for ~9000 movies]
  Liked: []
  --> No queries to send. Show random movies.

User likes "The Dark Knight":
  Model: [unchanged]
  Liked: ["The Dark Knight"]
  --> 1 query, but < 3 so still random

User likes "Inception":
  Model: [unchanged]
  Liked: ["The Dark Knight", "Inception"]
  --> 2 queries, but < 3 so still random

User likes "Interstellar":
  Model: [unchanged]
  Liked: ["The Dark Knight", "Inception", "Interstellar"]
  --> 3 queries! Now aggregating recs from all 3.
  --> Model is queried 3 times, results merged.

User likes "The Matrix":
  Model: [unchanged]
  Liked: ["The Dark Knight", "Inception", "Interstellar", "The Matrix"]
  --> 4 queries. Even richer aggregation.
  --> Movies similar to ALL FOUR liked films rise to the top.
```

### The Analogy

Imagine you walk into a bookstore and tell the clerk: "I liked Harry Potter."
The clerk suggests other fantasy books. Then you say: "I also liked Sherlock
Holmes." Now the clerk can triangulate -- you like stories with clever
protagonists, mystery elements, and British settings. The clerk's knowledge
(the "model") has not changed. But the _query_ has gotten richer, so the
suggestions get better.

MovieMatch works the same way. More likes = more queries to the same fixed
model = better aggregated results.

### This Is a "Session-Based" Strategy

In recommendation system terminology, this approach is:

- **Session-based:** Preferences only exist for the current session. Close the
  browser, lose your profile.
- **Non-learning:** The model does not update. It is a static lookup table of
  movie similarities.
- **Query-expansion:** Each new like adds another "query" whose results are
  merged with previous ones.

This is different from systems like Netflix or Spotify, which:

- Store your entire history permanently
- Periodically retrain models on millions of users' data
- Use collaborative filtering ("users like you also watched...")

---

## 7. Limitations and Possible Improvements

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

**4. Dislikes are wasted.** When a user swipes left, the movie is added to
`seen` so it will not reappear -- but the dislike is not used to _inform_ future
recommendations. A smarter system could:

- Penalize movies similar to disliked ones
- Learn "anti-preferences" (user dislikes horror --> reduce horror
  recommendations)
- Use dislikes to refine the similarity score

Currently, disliking "Saw" does not reduce the chance of seeing "Hostel" next.

**5. Cold start quality.** During Phase 1, popularity weighting helps, but
the user might still see movies they find irrelevant. There is no genre
preference survey, no "pick 5 movies you like" onboarding flow.

### Possible Improvements

| Improvement                     | Difficulty | Impact |
| ------------------------------- | ---------- | ------ |
| Per-user sessions (cookies)     | Low        | High   |
| Database persistence (SQLite)   | Low        | High   |
| Use dislikes as negative signal | Medium     | Medium |
| Onboarding genre picker         | Low        | Medium |
| Collaborative filtering         | High       | High   |
| Implicit signals (hover time)   | Medium     | Low    |
| A/B testing of algorithms       | High       | Medium |

---

## Summary

The MovieMatch application takes a pre-trained content-based recommendation
model and wraps it in an interactive feedback loop:

```
  +----------+     swipe      +-----------+     query      +-------+
  |  User    |  ----------->  |  Session  |  ----------->  | Model |
  |  (browser|                |  State    |                | (fixed|
  |   card)  |  <-----------  |  (liked,  |  <-----------  |  cos  |
  +----------+   next movie   |   seen)   |   recs list    |  sim) |
                               +-----------+                +-------+
```

The model never changes. The session state grows. As the session state grows,
the model is queried from more angles, and the aggregated results become
increasingly tailored to the user's taste. This is the central mechanism: a
fixed model, queried dynamically, producing progressively better results.
