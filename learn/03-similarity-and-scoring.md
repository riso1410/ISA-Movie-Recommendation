# 03 - Similarity, Scoring, and Diversity

This document explains the mathematical foundations behind how our movie
recommender decides which movies are "similar" to one another, how it balances
similarity with quality, and how it ensures recommendations are diverse rather
than repetitive. We build every concept from the ground up.

---

## Table of Contents

1. [Cosine Similarity -- from Scratch](#1-cosine-similarity----from-scratch)
2. [IMDB Weighted Rating](#2-imdb-weighted-rating)
3. [Combined Scoring](#3-combined-scoring)
4. [MMR -- Diversity Re-ranking](#4-mmr----diversity-re-ranking)
5. [The Full recommend() Flow](#5-the-full-recommend-flow)
6. [User Profile Vector](#6-user-profile-vector)
7. [Soft Penalty for Disliked Content](#7-soft-penalty-for-disliked-content)
8. [Exploration -- Breaking Echo Chambers](#8-exploration----breaking-echo-chambers)

---

## 1. Cosine Similarity -- from Scratch

### 1.1 The Core Idea: Movies as Arrows

Imagine every movie is described by a list of numbers. For example, we might
(very simplified) describe a movie with just two numbers:

- How much "action" it contains (0 to 10)
- How much "romance" it contains (0 to 10)

We can draw each movie as an **arrow** (a vector) on a 2D plane:

```
Romance
  ^
  |
10|                     * Movie C (2, 9)
  |                   /
  |                 /
  |               /
  |             /
 5|           *  Movie B (4, 5)
  |         /
  |       /         * Movie A (8, 3)
  |     /         /
  |   /         /
  | /         /
  +------*--/-------------------> Action
  0         5         10
```

Movie A (action=8, romance=3) is an action-heavy movie with some romance.
Movie C (action=2, romance=9) is a romance-heavy movie with little action.
Movie B (action=4, romance=5) sits somewhere in between.

The key insight: **the angle between two arrows tells us how similar two movies
are**. If two arrows point in nearly the same direction, the movies emphasize
the same features in similar proportions -- they are similar. If two arrows
point in very different directions, the movies are different.

```
         Movie C
          /
         / ) large angle = very different from Movie A
        /
       /
      /
     /_____________ Movie A

         Movie B
          /
         / ) small angle = somewhat similar to Movie A
        /___________  Movie A
```

**Cosine similarity** measures exactly this: the cosine of the angle between
two vectors.

### 1.2 The Formula

Given two vectors A and B, cosine similarity is:

```
                        A . B
cos(theta) = -------------------------
              ||A|| x ||B||
```

There are three pieces here. Let us define each one.

#### Dot Product (A . B)

The **dot product** multiplies corresponding elements and sums them up.

For two vectors A = [a1, a2, ..., an] and B = [b1, b2, ..., bn]:

```
A . B = a1*b1 + a2*b2 + ... + an*bn
```

Intuition: the dot product is large when both vectors have large values in the
same positions (they agree on what features are important).

#### Magnitude (||A||)

The **magnitude** (or length or norm) of a vector is:

```
||A|| = sqrt(a1^2 + a2^2 + ... + an^2)
```

This is just the Pythagorean theorem generalized to n dimensions. It tells us
"how long" the arrow is.

#### Putting It Together

We divide the dot product by the product of the magnitudes. This **normalizes**
the result so it does not depend on how long the vectors are -- only on the
angle between them.

### 1.3 Worked Example

Let us compute the cosine similarity between Movie A and Movie B from our
diagram.

```
Movie A = [8, 3]    (action=8, romance=3)
Movie B = [4, 5]    (action=4, romance=5)
```

**Step 1: Dot Product**

```
A . B = 8*4 + 3*5
      = 32  + 15
      = 47
```

**Step 2: Magnitudes**

```
||A|| = sqrt(8^2 + 3^2) = sqrt(64 + 9) = sqrt(73) = 8.544
||B|| = sqrt(4^2 + 5^2) = sqrt(16 + 25) = sqrt(41) = 6.403
```

**Step 3: Cosine Similarity**

```
cos(theta) = 47 / (8.544 x 6.403)
           = 47 / 54.72
           = 0.859
```

Result: **0.859** -- quite similar! Both movies lean toward action.

Now let us compute similarity between Movie A and Movie C:

```
Movie A = [8, 3]
Movie C = [2, 9]
```

```
A . C = 8*2 + 3*9 = 16 + 27 = 43

||A|| = 8.544   (same as before)
||C|| = sqrt(4 + 81) = sqrt(85) = 9.220

cos(theta) = 43 / (8.544 x 9.220)
           = 43 / 78.77
           = 0.546
```

Result: **0.546** -- moderately similar. They share some features but point in
quite different directions (action-heavy vs romance-heavy).

### 1.4 The Range: What the Numbers Mean

Cosine similarity always falls between 0 and 1 (for non-negative feature
vectors like TF-IDF weights, which are always >= 0):

```
  1.0  -- Identical direction. The movies share the exact same feature
           proportions. (They might differ in scale, but not in "what they
           are about.")

  0.7+ -- Very similar. Strong overlap in themes/cast/genre.

  0.3-0.7 -- Moderate similarity. Some shared features, but notable
              differences.

  0.0  -- Completely different. No features in common at all.
           (e.g., one movie's features are entirely absent in the other.)
```

Note: in general math, cosine similarity can range from -1 to 1, but because
TF-IDF and count vectors only contain non-negative values (word frequencies
cannot be negative), our similarities are always in [0, 1].

### 1.5 Why Cosine and Not Euclidean Distance?

You might wonder: why not just measure the straight-line distance between
two points (Euclidean distance)?

The answer is **magnitude invariance**. Consider these two movie reviews, both
about a horror comedy:

```
Short review  = [1, 0, 1, 0, 2]   (mentions "scary" once, "funny" twice)
Long review   = [3, 0, 3, 0, 6]   (mentions "scary" 3 times, "funny" 6 times)
```

These reviews talk about the **exact same topics** in the **exact same
proportions** -- the long one is just longer.

**Euclidean distance** between them:

```
d = sqrt((3-1)^2 + (0-0)^2 + (3-1)^2 + (0-0)^2 + (6-2)^2)
  = sqrt(4 + 0 + 4 + 0 + 16)
  = sqrt(24)
  = 4.90   <-- says they're quite far apart!
```

**Cosine similarity** between them:

```
dot product = 1*3 + 0*0 + 1*3 + 0*0 + 2*6 = 3 + 3 + 12 = 18
||short||   = sqrt(1+0+1+0+4) = sqrt(6)   = 2.449
||long||    = sqrt(9+0+9+0+36) = sqrt(54)  = 7.348

cosine = 18 / (2.449 * 7.348) = 18 / 18 = 1.0   <-- perfectly similar!
```

Cosine similarity correctly identifies that these two vectors point in the
**same direction** -- they represent the same mix of topics. The fact that one
is "longer" (the review has more words) does not matter. This is exactly the
behavior we want: a movie with a longer description should still be recognized
as similar to a movie with a shorter description, as long as they discuss the
same themes.

### 1.6 The Cosine Similarity Matrix

In our recommender we have N movies (around 9,000 in this dataset). After
converting every movie into a feature vector via TF-IDF, we compute the cosine
similarity between **every pair** of movies. This produces an N x N matrix:

```
              Movie_0   Movie_1   Movie_2   Movie_3  ...  Movie_N
Movie_0      [ 1.000    0.234     0.056     0.891   ...  0.012  ]
Movie_1      [ 0.234    1.000     0.445     0.102   ...  0.330  ]
Movie_2      [ 0.056    0.445     1.000     0.033   ...  0.678  ]
Movie_3      [ 0.891    0.102     0.033     1.000   ...  0.009  ]
  ...           ...      ...       ...       ...    ...   ...
Movie_N      [ 0.012    0.330     0.678     0.009   ...  1.000  ]
```

Key properties of this matrix:

- **Diagonal is always 1.0**: every movie is perfectly similar to itself.
- **Symmetric**: similarity(A, B) = similarity(B, A). The angle between A and
  B is the same as the angle between B and A.
- **Entry [i, j]** = cosine similarity between movie i and movie j.

In the code (`predict_model.py`, line 90):

```python
self.cosine_sim = cosine_similarity(self.feature_matrix)
```

This single call computes the entire N x N matrix. For ~9,000 movies, that is
about 81 million similarity values. This is computationally expensive (a few
seconds), but we do it **once at training time**. After that, looking up the
similarity between any two movies is just a matrix lookup -- instant.

### 1.7 What Our Vectors Actually Contain

In the simplified example above, our vectors had just 2 dimensions (action,
romance). In reality, each movie's vector is constructed by combining **8
different fields**, each processed by its own vectorizer:

```
Field       Vectorizer       Weight    What it captures
----------- ---------------- --------- --------------------------------
overview    TF-IDF (15000)   1.0       Plot description words
genres      Count             1.5       Genre labels (Action, Comedy...)
keywords    TF-IDF (5000)    1.2       Plot keywords from TMDb
cast        Count             1.0       Top cast member names
director    Count             2.0       Director name (highest weight!)
decade      Count             0.3       Release decade (1990s, 2000s...)
language    Count             0.5       Original language
collection  Count             1.5       Franchise (e.g., "Star Wars")
```

Each field is vectorized independently and then **horizontally stacked** into
one wide sparse matrix. The weights multiply each sub-matrix so that, for
example, a director match contributes 2x as much as a cast match:

```python
# From predict_model.py, lines 68-89
matrices = []
for field, (col, vec_cls, vec_kwargs) in FIELD_CONFIG.items():
    weight = self.weights.get(field, 0.0)
    ...
    matrix = vectorizer.fit_transform(texts)
    if weight != 1.0:
        matrix = matrix * weight
    matrices.append(matrix)

self.feature_matrix = hstack(matrices, format='csr')  # combine all fields
```

The final vector for each movie might have 20,000+ dimensions. Cosine
similarity works just as well in 20,000 dimensions as it does in 2 -- the
formula is identical, just with more terms in the sums.

---

## 2. IMDB Weighted Rating

### 2.1 The Problem

Imagine you are choosing a movie. You see:

```
Movie X: Average rating 10.0/10  (1 vote)
Movie Y: Average rating  8.5/10  (10,000 votes)
```

Which is actually the better movie? Almost certainly **Movie Y**. That single
voter for Movie X might have been the director's mother. With only 1 vote, we
have almost no confidence in the rating.

A naive approach (just sort by average rating) would rank Movie X above Movie Y.
We need a smarter formula that accounts for **how many people voted**.

### 2.2 The IMDB Weighted Rating Formula

IMDB developed a formula to solve exactly this problem:

```
           v                m
WR  =  ------- x R   +  ------- x C
         v + m             v + m
```

Where:

| Variable | Meaning                                      | In our code                        |
| -------- | -------------------------------------------- | ---------------------------------- |
| **v**    | Number of votes this movie received          | `vote_count` column                |
| **R**    | This movie's average rating (from voters)    | `vote_average` column              |
| **m**    | Minimum votes required to be considered      | 60th percentile of all vote counts |
| **C**    | Mean rating across ALL movies in the dataset | Mean of `vote_average` column      |

### 2.3 Understanding the Formula Intuitively

Look at the two "weight" fractions:

```
v/(v+m)  -- how much we trust THIS movie's actual rating
m/(v+m)  -- how much we fall back to the GLOBAL average
```

These two fractions **always add up to 1**:

```
v/(v+m) + m/(v+m) = (v+m)/(v+m) = 1
```

So the weighted rating is a **blend** between:

- **R** (the movie's own average rating)
- **C** (the global average rating of all movies)

The blend ratio depends on how many votes the movie has **relative to m**:

```
Few votes (v << m):
    v/(v+m) is close to 0,  m/(v+m) is close to 1
    WR is close to C (the global average)
    "We don't trust this movie's rating, fall back to average."

Many votes (v >> m):
    v/(v+m) is close to 1,  m/(v+m) is close to 0
    WR is close to R (the movie's own rating)
    "Lots of people voted, we trust this movie's actual rating."

v equals m:
    v/(v+m) = 0.5,  m/(v+m) = 0.5
    WR is exactly the midpoint of R and C
    "Half confidence -- split the difference."
```

Think of it as a **trust dial** that smoothly rotates from "trust the global
average" to "trust this movie's own rating" as votes increase.

### 2.4 Worked Example

Let us say our dataset has these statistics:

```
C (mean rating across all movies)        = 5.9
m (60th percentile of vote counts)       = 160
```

**Movie Alpha** -- A well-known blockbuster:

```
v = 8,000 votes
R = 7.8 average rating

WR = (8000 / (8000 + 160)) x 7.8 + (160 / (8000 + 160)) x 5.9
   = (8000 / 8160)         x 7.8 + (160 / 8160)          x 5.9
   = 0.9804                x 7.8 + 0.0196                 x 5.9
   = 7.647                       + 0.116
   = 7.763
```

With 8,000 votes, the formula barely adjusts the rating (7.8 -> 7.763). We
trust the crowd.

**Movie Beta** -- A small indie film:

```
v = 12 votes
R = 9.2 average rating

WR = (12 / (12 + 160)) x 9.2 + (160 / (12 + 160)) x 5.9
   = (12 / 172)         x 9.2 + (160 / 172)         x 5.9
   = 0.0698             x 9.2 + 0.9302               x 5.9
   = 0.642                    + 5.488
   = 6.130
```

Despite having a 9.2 average, Movie Beta's weighted rating is pulled down to
6.130 because only 12 people voted. We do not trust so few votes.

**Movie Gamma** -- A mediocre movie with tons of votes:

```
v = 5,000 votes
R = 5.5 average rating

WR = (5000 / 5160) x 5.5 + (160 / 5160) x 5.9
   = 0.969          x 5.5 + 0.031         x 5.9
   = 5.330                + 0.183
   = 5.513
```

Stays close to its own 5.5 rating. Many votes confirm it truly is mediocre.

### 2.5 Why 60th Percentile for m?

The threshold `m` is set to the **60th percentile** of all vote counts:

```python
# predict_model.py, line 48
self.m = self.smd['vote_count'].quantile(0.60)
```

This means 60% of all movies have fewer votes than `m`. Why this choice?

- **Too low** (e.g., 20th percentile): Almost every movie gets trusted,
  including those with very few votes. Noisy ratings dominate.
- **Too high** (e.g., 95th percentile): Almost every movie gets pulled toward
  the global average. Only mega-blockbusters keep their own ratings. We lose
  the ability to distinguish most movies.
- **60th percentile**: A moderate cutoff. Movies need a decent number of votes
  before we start trusting their ratings, but it is not so strict that only
  blockbusters get through.

### 2.6 Normalization to [0, 1]

After computing weighted ratings, we **normalize** them to a 0-to-1 range:

```python
# predict_model.py, lines 59-63
wr = self.smd['weighted_rating']
wr_min, wr_max = wr.min(), wr.max()
if wr_max > wr_min:
    self.smd['wr_norm'] = (wr - wr_min) / (wr_max - wr_min)
else:
    self.smd['wr_norm'] = 0.5
```

This is **min-max normalization**:

```
                  wr - wr_min
wr_norm  =  ---------------------
              wr_max - wr_min
```

For example, if weighted ratings range from 4.2 to 8.1:

```
A movie with WR = 4.2  ->  wr_norm = (4.2 - 4.2)/(8.1 - 4.2) = 0.0
A movie with WR = 8.1  ->  wr_norm = (8.1 - 4.2)/(8.1 - 4.2) = 1.0
A movie with WR = 6.15 ->  wr_norm = (6.15 - 4.2)/(8.1 - 4.2) = 0.5
```

Why normalize? Because in the next section we combine similarity scores
(already in [0, 1]) with weighted ratings. If we left ratings on their raw
scale (~4 to ~8), they would completely dominate the combined score. Normalizing
puts both quantities on the same [0, 1] scale so they can be blended fairly.

---

## 3. Combined Scoring

### 3.1 The Problem with Pure Similarity

If we recommend movies based **only** on cosine similarity, we might get
results like:

```
You liked: "The Dark Knight" (rating 9.0, 12,000 votes)

Top similar movies (similarity only):
1. "Batman: Gotham Knight"    sim=0.92  rating=6.2  votes=89
2. "Batman Unlimited"         sim=0.89  rating=5.1  votes=34
3. "The Dark Knight Returns"  sim=0.87  rating=7.1  votes=450
```

The top picks are highly similar (they share Batman keywords, cast, genre) but
some are obscure, low-quality direct-to-video releases. A user who loved The
Dark Knight probably wants high-quality recommendations, not just the most
topically similar ones.

### 3.2 The Combined Score Formula

We blend similarity with quality:

```
score = alpha x similarity + (1 - alpha) x wr_norm
```

Where:

- **similarity** is the cosine similarity (0 to 1)
- **wr_norm** is the normalized weighted rating (0 to 1)
- **alpha** controls the balance (default: **0.7**)

With alpha=0.7:

```
score = 0.7 x similarity + 0.3 x wr_norm
```

Similarity gets 70% of the influence, quality gets 30%. This means:

- Content relevance still matters most (you will not be recommended a random
  highly-rated movie that has nothing to do with your taste).
- But among similarly-relevant movies, higher-quality ones get a boost.

```python
# predict_model.py, lines 113-114
wr_norms = self.smd['wr_norm'].values[candidate_indices]
combined_scores = alpha * candidate_sims + (1 - alpha) * wr_norms
```

### 3.3 Worked Example

Suppose you liked "Inception" and we have three candidates:

```
                      Similarity   wr_norm
Movie P (obscure)        0.85        0.30
Movie Q (classic)        0.60        0.95
Movie R (good match)     0.72        0.80
```

With alpha=0.7:

```
Score(P) = 0.7 x 0.85 + 0.3 x 0.30 = 0.595 + 0.090 = 0.685
Score(Q) = 0.7 x 0.60 + 0.3 x 0.95 = 0.420 + 0.285 = 0.705
Score(R) = 0.7 x 0.72 + 0.3 x 0.80 = 0.504 + 0.240 = 0.744
```

**Ranking by combined score: R (0.744) > Q (0.705) > P (0.685)**

Without the quality adjustment (alpha=1.0, pure similarity):

```
Score(P) = 0.85,  Score(R) = 0.72,  Score(Q) = 0.60
Ranking: P > R > Q
```

The combined score reshuffled the ranking. Movie P was the most similar but
its low quality dragged it to third place. Movie R, which had good similarity
AND good quality, rose to first. Movie Q jumped from last to second because
its excellent quality compensated for moderate similarity.

### 3.4 Choosing Alpha

The alpha parameter lets you tune the relevance-vs-quality tradeoff:

```
alpha = 1.0  -->  Pure similarity. Ignore quality entirely.
alpha = 0.7  -->  Default. Similarity dominates, quality is a tiebreaker.
alpha = 0.5  -->  Equal weight to both.
alpha = 0.0  -->  Pure quality. Ignore similarity entirely (bad: just
                   recommends the highest-rated movies regardless of taste).
```

The default of 0.7 was chosen because the primary goal is to find movies that
match the user's taste (similarity), but among candidates with similar
relevance, we want to surface the better-reviewed ones.

---

## 4. MMR -- Diversity Re-ranking

### 4.1 The Problem: Recommendation Echo Chambers

Even with combined scoring, the top results can be repetitive. If you like
"The Avengers", the top 10 might be:

```
1. Avengers: Age of Ultron     (Marvel, superhero, ensemble)
2. Captain America: Civil War   (Marvel, superhero, ensemble)
3. Iron Man 3                   (Marvel, superhero)
4. Thor: Ragnarok               (Marvel, superhero)
5. Guardians of the Galaxy      (Marvel, superhero)
6. Ant-Man                      (Marvel, superhero)
7. Doctor Strange               (Marvel, superhero)
8. Spider-Man: Homecoming       (Marvel, superhero)
9. Black Panther                (Marvel, superhero)
10. Captain Marvel              (Marvel, superhero)
```

All Marvel! The user probably already knows about these. A better list might
include some non-Marvel action movies, sci-fi films, or ensemble comedies that
share **some** qualities with The Avengers but are not carbon copies.

### 4.2 The MMR Formula

**Maximal Marginal Relevance (MMR)** solves this by penalizing candidates that
are too similar to movies **already selected** for the recommendation list.

For each remaining candidate movie, we compute:

```
MMR_score = lambda x relevance  -  (1 - lambda) x max_sim_to_already_selected
```

Where:

- **relevance** = the combined score from Section 3 (similarity + quality)
- **max_sim_to_already_selected** = the highest cosine similarity between this
  candidate and any movie we have ALREADY placed in the recommendation list
- **lambda** = controls the relevance-diversity tradeoff (default: 0.5)

The key insight: if a candidate is very similar to something already in our
list, `max_sim_to_already_selected` is high, which **subtracts** from its MMR
score. This pushes it down the ranking in favor of candidates that bring
something new.

### 4.3 The Greedy Selection Algorithm

MMR works as a **greedy iterative** process. We do not score all candidates at
once. Instead, we build the recommendation list one movie at a time:

```
GIVEN: candidate_pool (e.g., 50 movies), want to select n (e.g., 10)

Step 1: Pick the candidate with the HIGHEST combined score.
        (First pick = best overall, no diversity penalty yet because
         the selected list is empty.)

Step 2: For each remaining candidate, compute:
        MMR = lambda * combined_score - (1-lambda) * max_sim(candidate, selected_set)
        Pick the candidate with the highest MMR.

Step 3: Repeat Step 2 until we have n movies.
```

Each iteration, the `max_sim_to_already_selected` term changes because the
selected set grows. A candidate that was fine after picking movie 1 might get
penalized after picking movie 2 (if movie 2 happens to be very similar to it).

Here is the corresponding code:

```python
# predict_model.py, lines 132-165
def _mmr_rerank(self, candidate_indices, scores, n, lam):
    candidate_sim = self.cosine_sim[np.ix_(candidate_indices, candidate_indices)]
    remaining = list(range(len(candidate_indices)))
    selected = []

    for _ in range(min(n, len(candidate_indices))):
        if not selected:
            # First pick: highest combined score
            best_local = max(remaining, key=lambda i: scores[i])
        else:
            best_val = -float('inf')
            best_local = remaining[0]
            for i in remaining:
                max_sim_to_selected = max(candidate_sim[i][j] for j in selected)
                mmr_score = lam * scores[i] - (1 - lam) * max_sim_to_selected
                if mmr_score > best_val:
                    best_val = mmr_score
                    best_local = i
        selected.append(best_local)
        remaining.remove(best_local)
```

### 4.4 Lambda: The Diversity Dial

```
lambda = 1.0  -->  Pure relevance. MMR = 1.0 * score - 0.0 * penalty
                   No diversity at all. Same as no MMR.

lambda = 0.5  -->  Default. Equal weight to relevance and diversity.
                   Balanced recommendations.

lambda = 0.0  -->  Pure diversity. MMR = 0.0 * score - 1.0 * penalty
                   Picks the MOST DIFFERENT movies possible. Ignores
                   relevance entirely (bad: random-seeming results).
```

The default lambda=0.5 means: "I care equally about recommending relevant
movies and recommending diverse movies."

### 4.5 Concrete MMR Walkthrough

Let us walk through selecting 3 movies from 5 candidates using MMR with
lambda=0.5.

**Setup**: You liked "The Matrix". Here are 5 candidates with their combined
scores and pairwise similarities to each other:

```
Candidates and their combined scores:
  A: "Matrix Reloaded"        score = 0.90
  B: "Matrix Revolutions"     score = 0.85
  C: "Inception"              score = 0.75
  D: "Blade Runner"           score = 0.70
  E: "John Wick"              score = 0.65

Pairwise similarities between candidates:
         A      B      C      D      E
  A   [ 1.00   0.92   0.35   0.30   0.40 ]
  B   [ 0.92   1.00   0.33   0.28   0.38 ]
  C   [ 0.35   0.33   1.00   0.45   0.25 ]
  D   [ 0.30   0.28   0.45   1.00   0.20 ]
  E   [ 0.40   0.38   0.25   0.20   1.00 ]
```

Note: A and B are extremely similar to each other (0.92) -- they are both
Matrix sequels. C and D share some sci-fi themes (0.45). E is fairly
independent.

**Iteration 1: Pick the highest combined score.**

```
Selected: []  (empty)
Pick: A ("Matrix Reloaded") with score 0.90
```

Selected so far: **[A]**

**Iteration 2: Compute MMR for each remaining candidate.**

For each candidate, we find its maximum similarity to any already-selected
movie (currently just A):

```
Candidate B:
  max_sim_to_selected = sim(B, A) = 0.92
  MMR = 0.5 * 0.85 - 0.5 * 0.92 = 0.425 - 0.460 = -0.035

Candidate C:
  max_sim_to_selected = sim(C, A) = 0.35
  MMR = 0.5 * 0.75 - 0.5 * 0.35 = 0.375 - 0.175 = 0.200

Candidate D:
  max_sim_to_selected = sim(D, A) = 0.30
  MMR = 0.5 * 0.70 - 0.5 * 0.30 = 0.350 - 0.150 = 0.200

Candidate E:
  max_sim_to_selected = sim(E, A) = 0.40
  MMR = 0.5 * 0.65 - 0.5 * 0.40 = 0.325 - 0.200 = 0.125
```

```
MMR scores:  B = -0.035,  C = 0.200,  D = 0.200,  E = 0.125
```

**B gets a negative MMR score!** Even though it had the second-highest combined
score (0.85), it is so similar to already-selected A (0.92 similarity) that the
diversity penalty kills it. The algorithm picks C or D (tied at 0.200). Let us
say it picks **C** (first encountered).

Selected so far: **[A, C]**

**Iteration 3: Compute MMR for remaining candidates (B, D, E).**

Now `max_sim_to_selected` checks similarity to BOTH A and C:

```
Candidate B:
  max_sim = max(sim(B,A), sim(B,C)) = max(0.92, 0.33) = 0.92
  MMR = 0.5 * 0.85 - 0.5 * 0.92 = 0.425 - 0.460 = -0.035

Candidate D:
  max_sim = max(sim(D,A), sim(D,C)) = max(0.30, 0.45) = 0.45
  MMR = 0.5 * 0.70 - 0.5 * 0.45 = 0.350 - 0.225 = 0.125

Candidate E:
  max_sim = max(sim(E,A), sim(E,C)) = max(0.40, 0.25) = 0.40
  MMR = 0.5 * 0.65 - 0.5 * 0.40 = 0.325 - 0.200 = 0.125
```

```
MMR scores:  B = -0.035,  D = 0.125,  E = 0.125
```

B is still penalized. D and E are tied; let us say it picks **D**.

**Final selection: [A, C, D]**

```
Without MMR (pure score ranking):  A, B, C
With MMR:                          A, C, D
```

MMR replaced "Matrix Revolutions" (a near-duplicate of "Matrix Reloaded")
with "Blade Runner" (a different kind of sci-fi). The result is a more
**interesting and varied** set of recommendations while still being relevant.

---

## 5. The Full `recommend()` Flow

Now let us trace the complete `recommend()` method from start to finish.
This is the code at lines 93-130 of `predict_model.py`.

### Step-by-Step Walkthrough

```
Input: recommend(title="The Dark Knight", n=10, alpha=0.7, mmr_lambda=0.5)
```

**Step 1: Look up the movie index.**

```python
idx = self.indices[title]   # e.g., idx = 4237
```

The `indices` series maps movie titles to their row position in the DataFrame.

**Step 2: Get the candidate pool (top n\*5 by raw similarity).**

```python
n_candidates = n * 5           # 10 * 5 = 50 candidates
sim_scores = self.cosine_sim[idx]   # Row 4237 of the NxN matrix
candidate_indices = np.argsort(sim_scores)[::-1][1:n_candidates + 1]
candidate_sims = sim_scores[candidate_indices]
```

This retrieves row 4237 from the similarity matrix -- a vector of ~9,000
similarity scores (one per movie). We sort them in descending order and take
the top 50 (skipping position 0, which is the movie itself with similarity
1.0).

Why 5x over-fetch? Because MMR will re-rank and potentially push some high-
similarity candidates down. We need a large enough pool to find diverse
alternatives.

```
Visualization of candidate selection:

All ~9000 movies sorted by similarity to "The Dark Knight":
|<-- top 50 candidates -->|<--- remaining ~8950 (ignored) --->|
[0.95, 0.91, 0.88, ..., 0.42, | 0.41, 0.40, ..., 0.00]
```

**Step 3: Compute combined scores.**

```python
wr_norms = self.smd['wr_norm'].values[candidate_indices]
combined_scores = alpha * candidate_sims + (1 - alpha) * wr_norms
```

For each of the 50 candidates:

```
combined_score = 0.7 x cosine_similarity + 0.3 x normalized_weighted_rating
```

This blends content relevance with quality (see Section 3).

**Step 4: MMR re-rank for diversity.**

```python
selected = self._mmr_rerank(candidate_indices, combined_scores, n, mmr_lambda)
```

From the 50 candidates, greedily select 10 movies that balance high combined
scores with diversity (see Section 4). Returns a list of (index, score) tuples.

```
50 candidates  -->  MMR greedy selection  -->  10 diverse recommendations
```

**Step 5: Build the result DataFrame.**

```python
result_indices = [s[0] for s in selected]
result_scores = [s[1] for s in selected]
result_sims = [sim_scores[i] for i in result_indices]

result = self.smd[['title', 'genres', 'vote_average', 'vote_count',
                    'weighted_rating']].iloc[result_indices].copy()
result['similarity_score'] = result_sims
result['combined_score'] = result_scores
return result.reset_index(drop=True)
```

The final output is a DataFrame like:

```
   title                     genres              vote_avg  vote_cnt  WR    sim    combined
0  The Dark Knight Rises     Action|Crime|Drama  7.6       9106      7.58  0.91   0.82
1  Inception                 Action|Sci-Fi       8.1       14075     8.05  0.68   0.73
2  Batman Begins             Action|Crime|Drama  7.5       7511      7.48  0.88   0.78
3  Interstellar              Adventure|Drama     8.1       11187     8.04  0.52   0.60
4  The Prestige              Drama|Mystery       8.0       8235      7.96  0.55   0.62
...
```

Notice how the list is NOT sorted purely by similarity. Movie 1 (Inception) has
lower similarity than Movie 2 (Batman Begins) but its higher quality and the
MMR diversity reranking changed the ordering.

### Summary: The Complete Pipeline

```
                    +------------------+
                    |  Input: Title    |
                    +--------+---------+
                             |
                             v
                 +------------------------+
                 | Look up row in cosine  |
                 | similarity matrix      |
                 +--------+---------------+
                          |
                          v
               +--------------------------+
               | Take top 50 candidates   |
               | (n x 5 over-fetch)       |
               +--------+-----------------+
                         |
                         v
              +----------------------------+
              | Compute combined scores    |
              | 0.7*sim + 0.3*wr_norm      |
              +--------+-------------------+
                        |
                        v
              +----------------------------+
              | MMR re-rank (greedy)       |
              | Pick 10 diverse results    |
              +--------+-------------------+
                        |
                        v
              +----------------------------+
              | Return DataFrame with      |
              | titles, scores, metadata   |
              +----------------------------+
```

### Key Design Decisions Recap

| Decision                          | Why                                                                      |
| --------------------------------- | ------------------------------------------------------------------------ |
| Cosine similarity (not Euclidean) | Magnitude-invariant: focuses on feature proportions, not absolute counts |
| Precomputed NxN matrix            | Expensive once at training (~seconds), but instant lookups at query time |
| IMDB weighted rating              | Prevents low-vote movies from dominating via unreliable high averages    |
| 60th percentile for m             | Moderate trust threshold: not too permissive, not too strict             |
| Combined score (alpha=0.7)        | Similarity matters most, but quality breaks ties and filters junk        |
| MMR re-ranking (lambda=0.5)       | Prevents recommendation lists full of near-duplicates                    |
| 5x over-fetch for candidates      | Gives MMR enough room to find diverse alternatives                       |

---

## 6. User Profile Vector

Sections 1-5 describe the `recommend()` method, which takes a **single movie
title** as input and returns similar movies. This works well for one-off
queries, but in the live application users like and dislike _many_ movies over
a session. How do we combine all that feedback into a single recommendation?

### 6.1 The Problem with Per-Title Aggregation

The original approach (described in doc 05, Section 4) called `recommend()`
once per liked movie and merged the results:

```
User liked 5 movies
  → 5 separate recommend() calls
  → 5 × 15 = 75 candidate rows
  → Merge scores, deduplicate, sort
```

This has two weaknesses:

1. **Dislikes are ignored.** A user who dislikes horror never tells the model
   "less of this." The disliked movie is simply skipped and never shown again,
   but similar horror movies keep appearing.
2. **All likes are equal.** A deliberate "like" on a curated recommendation
   carries the same weight as a quick swipe-right on a random movie the user
   barely recognized.

### 6.2 The Profile Vector Concept

Instead of querying the model N times, we build **one vector** that represents
the user's overall taste, then compare it against every movie in the catalog in
a single pass.

The core idea: each movie is already a sparse feature vector (from TF-IDF and
Count vectorizers — see doc 02). If we take a **weighted sum** of the feature
vectors of all movies the user interacted with, we get a "user profile" in the
same feature space as the movies.

```
Profile = Σ  feature_vector[movie_i] × feedback_weight_i
         i∈swipe_log
```

### 6.3 Feedback Weights

Not all feedback is equal. The weight depends on two factors:

- **Direction**: right (like) or left (dislike)
- **Source**: where the user encountered the movie

```
(direction, source)         Weight    Rationale
-----------------------------------------------------------------
("right", "random")          1.0      Baseline: liked a random movie
("left",  "random")         -0.3      Weak negative: user may not know it
("right", "model")           2.5      Liked a model pick — strong signal
("left",  "model")          -1.2      Disliked a model pick — model was wrong
("right", "rec_list")        3.0      Liked from rec list — most intentional
("left",  "rec_list")       -1.5      Disliked from rec list — strongest neg
```

**Why the asymmetry?** Negative weights have smaller magnitude than positive
because dislikes are noisier. A user might swipe left simply because they have
already seen the movie, or because the poster looked unfamiliar. A swipe right
is a more confident signal — the user actively wants to watch it.

**Why source matters?** Liking a random movie you happened to see is a weaker
signal than deliberately clicking "like" on a curated recommendation in the
sidebar. The rec-list action is the most intentional (the user evaluated the
movie without being forced to), so it gets the highest weight.

### 6.4 Building the Profile: Step by Step

Let us walk through a concrete example. The user has made 4 swipes:

```
Swipe 1: "The Dark Knight"     → right, random      weight = +1.0
Swipe 2: "Scary Movie"         → left,  random      weight = -0.3
Swipe 3: "Inception"           → right, model       weight = +2.5
Swipe 4: "The Prestige"        → right, rec_list    weight = +3.0
```

Each movie has a sparse feature vector (say, 20,000 dimensions). We compute:

```
profile = TDK_vector * 1.0
        + Scary_vector * (-0.3)
        + Inception_vector * 2.5
        + Prestige_vector * 3.0
```

The resulting profile vector is strong in dimensions corresponding to "Nolan",
"thriller", "crime", "drama", "twist" — features shared by the three liked
movies. The horror/comedy dimensions from "Scary Movie" are subtracted, making
horror features slightly negative.

In the code (`app/main.py`, lines 259-266):

```python
for entry in log:
    idx = _movie_id_to_idx(entry["movie_id"])
    if idx is None:
        continue
    weight = FEEDBACK_WEIGHTS.get((entry["direction"], entry["source"]), 0.0)
    if weight == 0.0:
        continue
    profile = profile + recommender.feature_matrix[idx] * weight
```

### 6.5 L2 Normalization

After summing, the profile vector can have very large values (especially after
many swipes). We **L2-normalize** it — divide by its own magnitude — so that
its length becomes 1:

```
                    profile
profile_norm = ─────────────────
                ||profile||₂
```

Where `||profile||₂ = sqrt(sum of squared elements)`.

Why? Cosine similarity is scale-invariant (Section 1.5), but normalizing
explicitly ensures numerical stability and makes the profile comparable to the
already-normalized movie vectors in the feature matrix.

```python
# app/main.py, lines 271-274
norm = float(profile.multiply(profile).sum() ** 0.5)
if norm > 0:
    profile = profile / norm
```

Edge case: if `profile.nnz == 0` (all weights cancelled out — e.g., user liked
and disliked identical movies), we return an empty list. There is nothing to
recommend against a zero vector.

### 6.6 Scoring the Entire Catalog

With the normalized profile in hand, we compute cosine similarity between the
profile and **every movie** in one call:

```python
# app/main.py, line 277
cosine_scores = cosine_similarity(profile, recommender.feature_matrix).flatten()
```

This yields a 1D array of ~4,600 scores — one per movie. Then the familiar
combined scoring formula (Section 3):

```
combined = 0.7 × cosine_score + 0.3 × wr_norm
```

After excluding seen movies (set to `-inf`) and applying the soft penalty
(Section 7), the top candidates go through MMR re-ranking (Section 4) for
diversity.

### 6.7 Profile Vector vs Per-Title Aggregation

```
Per-Title Aggregation:                Profile Vector:

User liked 5 movies                   User made 12 swipes (5 right, 7 left)
  → 5 × recommend()                    → 1 weighted vector sum
  → Only likes used                    → Likes AND dislikes contribute
  → Equal weight per like              → Source-aware weighting
  → Results merged by score sum        → Cosine sim vs entire catalog
  → MMR per-title, then merge          → Single MMR pass over top-100
  → O(5 × N) similarity lookups        → O(1 × N) + O(N_features) sparse sum
```

The profile approach is both more expressive (uses negative feedback, variable
weights) and more efficient (single similarity computation instead of multiple
recommend() calls).

### 6.8 Performance

Building the profile vector is O(S × F) where S = number of swipes and
F = number of features — but since the feature matrix is sparse (~20k columns
but ~99.5% zeros), the actual work is proportional to non-zero entries.
The `cosine_similarity(1×F, N×F)` call for ~4,600 movies takes under 50ms on
a modern laptop. The entire `_profile_recommendations()` call completes in
under 100ms.

---

## 7. Soft Penalty for Disliked Content

### 7.1 The Idea

Section 6 showed how dislikes contribute negative weight to the profile vector.
But there is a subtlety: a movie can be far from the disliked movie in overall
feature space yet share specific problematic features (e.g., same director the
user hates). The profile vector handles this implicitly through the feature
weights, but we add an **explicit penalty** as a second line of defense.

### 7.2 How It Works

After computing combined scores for all movies, we check: "Is this movie very
similar to something the user disliked?"

Using the precomputed N×N cosine similarity matrix (Section 1.6):

```python
# app/main.py, lines 288-299
disliked_indices = [row_index for each disliked movie_id]

max_sim_to_disliked = np.max(
    recommender.cosine_sim[:, disliked_indices], axis=1
)
penalty_mask = (max_sim_to_disliked > 0.7) & ~seen_mask
combined[penalty_mask] *= 0.5
```

For every movie in the catalog, we find the **maximum** cosine similarity
between that movie and any disliked movie. If this maximum exceeds **0.7**
(very similar), the movie's combined score is halved.

### 7.3 Worked Example

User disliked "Saw" (horror, torture, gory) and "Hostel" (horror, torture,
gory).

```
Movie: "The Texas Chain Saw Massacre"
  sim to "Saw"    = 0.82  ← above 0.7
  sim to "Hostel" = 0.75  ← above 0.7
  max_sim = 0.82 > 0.7
  → combined_score *= 0.5  (halved!)

Movie: "Get Out"
  sim to "Saw"    = 0.35
  sim to "Hostel" = 0.28
  max_sim = 0.35 < 0.7
  → No penalty (it's a different kind of horror)

Movie: "Inception"
  sim to "Saw"    = 0.05
  sim to "Hostel" = 0.03
  max_sim = 0.05 < 0.7
  → No penalty (completely different genre)
```

### 7.4 Why 0.7 and 0.5?

**Threshold 0.7**: In our similarity matrix, 0.7+ means very strong overlap —
typically same franchise, same director, or very similar genre+cast+keywords.
This is aggressive enough to catch near-duplicates of disliked content without
penalizing movies that merely share one genre.

**Factor 0.5**: Halving (not zeroing) keeps the movie in the candidate pool.
If the user's profile otherwise strongly favors this movie, it can still
appear, just ranked lower. This prevents over-correction from a single dislike.

### 7.5 Two Layers of Negative Feedback

The system uses dislikes in **two complementary ways**:

1. **Profile vector** (Section 6): Disliked movies subtract their feature
   vectors from the profile, reducing similarity to movies with similar
   features. This is a _global, continuous_ signal.

2. **Soft penalty** (this section): Movies too similar to any specific disliked
   movie get their score halved. This is a _local, threshold-based_ safety net.

Together, these create robust negative feedback. The profile vector handles the
general direction ("less horror"), while the soft penalty handles the specific
cases ("definitely not this exact type of horror").

---

## 8. Exploration -- Breaking Echo Chambers

### 8.1 The Problem

Even with MMR diversity (Section 4), a user who consistently likes the same
type of movie will see an increasingly narrow slice of the catalog. The profile
vector reinforces itself: like Nolan → see more Nolan → like more Nolan →
profile becomes 90% Nolan features → only Nolan-like movies appear.

This is a **filter bubble** or **echo chamber**.

### 8.2 Epsilon-Greedy Exploration

The solution borrows from reinforcement learning: **epsilon-greedy** strategy.
With probability epsilon (ε = 0.15 = 15%), we ignore the model entirely and
show a random popularity-weighted movie instead.

```python
# app/main.py, lines 431-438
if random.random() < EXPLORATION_RATE:  # EXPLORATION_RATE = 0.15
    # Show a random movie (popularity-weighted)
    unseen = movies_df[~movies_df["id"].isin(seen)]
    weights = unseen["popularity"].clip(lower=0.1)
    weights = weights / weights.sum()
    chosen = unseen.sample(n=1, weights=weights)
    return movie_to_dict(chosen.iloc[0], source="random")
```

85% of the time (exploitation): the model shows its best recommendation.
15% of the time (exploration): a random popular movie breaks the pattern.

### 8.3 Why This Works

The random movie might be:

- **A genre the user hasn't tried**: They might discover they love documentaries
  even though they have only been swiping on action movies.
- **A disconfirming signal**: If the user dislikes the random pick, it adds
  negative weight to the profile, helping the model learn what to avoid.
- **A confirming signal**: If the user likes it, the profile broadens, leading
  to more diverse future recommendations.

### 8.4 Why 15%?

This is a practical balance:

```
ε = 0%   →  Pure exploitation. Echo chamber guaranteed.
ε = 5%   →  Very rare exploration. Mostly same bubble.
ε = 15%  →  Roughly 1 in 7 movies is random. Noticeable variety
             without feeling random.
ε = 50%  →  Half random. System feels broken — "why is it showing
             me movies I didn't ask for?"
```

At 15%, a user who swipes 20 movies will see about 3 random picks. This is
enough to occasionally break the pattern without undermining trust in the
recommendation system.

### 8.5 Exploration Only in Model Phase

Exploration is only active after 3+ likes (when the model phase begins). During
cold start (< 3 likes), all movies are already random — there is no bubble to
break.

```
                    User requests next movie
                              |
                              v
                   +----------------------+
                   | Liked count >= 3?    |
                   +----------------------+
                      |              |
                    No             Yes
                      |              |
                      v              v
               Random movie   Roll dice: random() < 0.15?
               (cold start)      |              |
                               Yes            No
                                 |              |
                                 v              v
                          Random movie    Profile-based
                          (exploration)   recommendation
```

---

## Summary of the Full Scoring Pipeline

The complete pipeline, combining all concepts from this document:

```
┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING TIME (once)                                                │
│                                                                      │
│  Per-field TF-IDF/Count → weighted sparse feature matrix → cosine   │
│  similarity N×N matrix → IMDB weighted ratings → wr_norm [0,1]      │
│                                                                      │
│  Stored in: recommender.pkl                                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  QUERY TIME (per request)                                            │
│                                                                      │
│  1. Build user profile: Σ feature_matrix[i] × feedback_weight       │
│  2. L2-normalize profile                                             │
│  3. Cosine similarity: profile vs all N movies                       │
│  4. Combined score: 0.7 × cosine + 0.3 × wr_norm                   │
│  5. Hard exclude: seen movies → -inf                                 │
│  6. Soft penalty: sim > 0.7 to disliked → score × 0.5              │
│  7. MMR re-rank top candidates for diversity                         │
│  8. Return top-N diverse, high-quality, taste-matching movies        │
└─────────────────────────────────────────────────────────────────────┘
```

| Decision                          | Why                                                                      |
| --------------------------------- | ------------------------------------------------------------------------ |
| Cosine similarity (not Euclidean) | Magnitude-invariant: focuses on feature proportions, not absolute counts |
| Precomputed N×N matrix            | Expensive once at training, instant lookups at query time                |
| IMDB weighted rating              | Prevents low-vote movies from dominating via unreliable high averages    |
| Combined score (alpha=0.7)        | Similarity matters most, quality breaks ties                             |
| MMR re-ranking (lambda=0.5)       | Prevents recommendation lists full of near-duplicates                    |
| User profile vector               | Single representation of all feedback, not per-title queries             |
| Asymmetric feedback weights       | Dislikes are noisier than likes, so carry less magnitude                 |
| Source-aware weights              | Intentional rec-list feedback > passive swipe on random                  |
| Soft penalty for disliked content | Second defense layer beyond profile vector subtraction                   |
| 15% exploration rate              | Epsilon-greedy prevents filter bubbles without breaking UX               |
