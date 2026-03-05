# 06 - Evaluating a Recommendation System

## Table of Contents

1. [Why Evaluate a Recommendation System?](#1-why-evaluate-a-recommendation-system)
2. [Precision@K](#2-precisionk)
3. [NDCG@K (Normalized Discounted Cumulative Gain)](#3-ndcgk-normalized-discounted-cumulative-gain)
4. [Coverage](#4-coverage)
5. [Intra-List Diversity (ILD)](#5-intra-list-diversity-ild)
6. [Novelty](#6-novelty)
7. [Serendipity@K](#7-serendipityk)
8. [Mean Reciprocal Rank (MRR)](#8-mean-reciprocal-rank-mrr)
9. [Per-Genre Precision](#9-per-genre-precision)
10. [Grid Search for Weight Optimization](#10-grid-search-for-weight-optimization)
11. [The Evaluation Procedure](#11-the-evaluation-procedure)
12. [Actual Results from Our System](#12-actual-results-from-our-system)

---

## 1. Why Evaluate a Recommendation System?

When you build a recommendation system, the first question is: **how do you know if it is any good?**

This is harder to answer than it looks. With a spam filter, there is a clear right answer: an email is either spam or not. With recommendations, "good" is subjective. A user might want something similar to what they liked, or something surprising they would never have found on their own. These are different -- and sometimes conflicting -- goals.

There are several dimensions of quality, and no single number captures all of them:

### Accuracy: Are the recommendations relevant?

The most obvious question. If you liked The Dark Knight, does the system recommend other crime thrillers, or does it suggest a romantic comedy? Accuracy metrics measure how well the recommended items match what we think the user wants.

### Diversity: Are the recommendations varied?

Imagine the system recommends 10 movies, and all 10 are Marvel superhero films. Even if you like Marvel movies, this list is not very useful -- you probably already know about most of them, and you might want some variety. Diversity measures how different the items within a single recommendation list are from each other.

### Coverage: Does the system use its full catalog?

A movie catalog might contain 9,000 films, but if the system only ever recommends the same 200 popular blockbusters, it is wasting most of its catalog. Coverage measures what fraction of the total catalog ever gets recommended.

### Novelty: Does it recommend surprising items?

Recommending Avengers: Endgame to a Marvel fan is "accurate" but not helpful -- they already know about it. Novelty measures whether the system surfaces lesser-known items that the user might not have discovered on their own.

### These goals conflict with each other

This is the key insight. Consider the tradeoffs:

```
                High Accuracy
                     |
    "Safe but boring" |  "The ideal"
    (all popular      |  (relevant AND
     blockbusters)    |   surprising)
                      |
  ----Low Novelty-----+-----High Novelty----
                      |
    "Random garbage"  |  "Interesting but
                      |   irrelevant"
                      |
                Low Accuracy
```

- Popular items are "safe" recommendations (high accuracy) but not novel.
- Obscure items are novel but risky (might not be relevant).
- A list of 10 nearly identical movies is very "accurate" but not diverse.
- A diverse list might include items that are less relevant individually.

A good evaluation framework measures ALL of these dimensions, so you can understand the tradeoffs your system makes.

---

## 2. Precision@K

### What it measures

Precision@K answers the question: **Of the K items I recommended, how many were actually relevant?**

This is the most intuitive accuracy metric. You ask for 10 recommendations, and you count how many of them are "good."

### The relevance problem

But wait -- how do we know if a recommendation is "relevant"? In a real system with real users, you could ask them: "Did you like this recommendation?" But in offline evaluation (no real users), we need a **proxy** for relevance.

This system uses **genre overlap** as the relevance proxy. The logic is:

> If you liked an action/sci-fi movie, then a recommendation that is also action or sci-fi is "relevant" (it shares at least one genre with the movie you started from).

This is not perfect -- a user who liked The Dark Knight might not care about every action movie ever made -- but it is a reasonable approximation that we can compute automatically.

### The formula

```
                    |relevant items in top K|
Precision@K  =  ─────────────────────────────
                            K
```

Where:

- K = number of recommendations (in our system, K = 10)
- "relevant" = the recommended movie shares at least one genre with the query movie

### Worked example

Suppose we ask for recommendations based on **The Dark Knight** (genres: Action, Crime, Drama, Thriller). We get 10 recommendations:

```
 #  | Recommended Movie          | Genres                         | Shares genre?
----+----------------------------+--------------------------------+--------------
 1  | The Dark Knight Rises      | Action, Crime, Drama, Thriller | YES (4 overlap)
 2  | Scarface                   | Action, Crime, Drama, Thriller | YES (4 overlap)
 3  | The Prestige               | Drama, Mystery, Thriller       | YES (2 overlap)
 4  | Training Day               | Action, Crime, Drama, Thriller | YES (4 overlap)
 5  | Thursday                   | Drama, Action, Crime, Thriller | YES (4 overlap)
 6  | Heat                       | Action, Crime, Drama, Thriller | YES (4 overlap)
 7  | My Neighbor Totoro         | Animation, Family, Fantasy     | NO  (0 overlap)
 8  | Running Scared             | Action, Crime, Drama, Thriller | YES (4 overlap)
 9  | The Asphalt Jungle         | Action, Crime, Drama, Thriller | YES (4 overlap)
10  | Bridget Jones's Diary      | Comedy, Romance                | NO  (0 overlap)
```

Count relevant items: 8 out of 10.

```
Precision@10 = 8 / 10 = 0.80
```

Interpretation: 80% of the recommendations share at least one genre with the query movie.

### How the code works

From `evaluate_model.py`, the `precision_at_k` function:

```python
def precision_at_k(recommender, test_movies, k=10):
    precisions = []

    for _, row in test_movies.iterrows():
        title = row['title']
        true_genres = row['genres']     # genres of the query movie

        recs = recommender.recommend(title, n=k)

        relevant = 0
        for _, rec_row in recs.iterrows():
            rec_genres = rec_row['genres']
            if _genre_overlap(true_genres, rec_genres) > 0:  # at least 1 shared genre
                relevant += 1

        precisions.append(relevant / k)

    return np.mean(precisions)   # average across all test movies
```

The function loops over every test movie, gets K recommendations for each, counts how many are relevant, computes precision for that query, and finally averages across all queries.

### Limitations

- Genre overlap is a **rough proxy**. Two action movies can be wildly different (The Dark Knight vs. Kung Fu Panda -- both "Action").
- It is **binary**: a movie is either relevant or not. There is no notion of "how relevant" (but NDCG fixes this -- see next section).
- It does not care about **order**: a relevant item at position 10 counts the same as one at position 1.

---

## 3. NDCG@K (Normalized Discounted Cumulative Gain)

### Intuition

Precision@K counts how many recommendations are relevant, but it ignores **where** they appear in the list. Consider two recommendation lists for an action movie:

```
List A:  [Action, Action, Action, Comedy, Comedy, Comedy, Comedy, Comedy, Comedy, Comedy]
List B:  [Comedy, Comedy, Comedy, Comedy, Comedy, Comedy, Comedy, Action, Action, Action]
```

Both have Precision@10 = 3/10 = 0.30. But List A is clearly better -- the relevant items are at the top, where the user sees them first. Most users never scroll past the first few results.

NDCG@K solves this by giving **more credit to relevant items at higher positions**.

### Step 1: Relevance scores (not just binary)

Unlike Precision@K which uses binary relevance (relevant or not), NDCG uses a **graded relevance score**. In our system, the relevance of a recommendation is the **number of overlapping genres** with the query movie.

Example with query movie genres = {Action, Crime, Drama, Thriller}:

```
Recommended movie genres:          | Overlapping genres       | Relevance
{Action, Crime, Drama, Thriller}   | All 4 overlap            | 4
{Drama, Mystery, Thriller}         | Drama, Thriller          | 2
{Comedy, Romance}                  | None                     | 0
{Action, Animation}                | Action                   | 1
```

Higher relevance = better match. A movie that shares 4 genres is more relevant than one that shares only 1.

### Step 2: Discounted Cumulative Gain (DCG)

DCG adds up relevance scores, but **discounts** items at lower positions using a logarithmic penalty:

```
              K
             ___
             \      relevance_i
DCG@K  =    /    ─────────────────
             ‾‾‾   log2(i + 1)
             i=1
```

The `log2(i + 1)` in the denominator is the **discount factor**. It makes items at position 1 count fully, while items further down the list count less and less.

Here is what the discount factor looks like for each position:

```
Position (i) | log2(i + 1) | Discount = 1/log2(i+1)
─────────────+─────────────+────────────────────────
      1      |    1.000    |    1.000   (full credit)
      2      |    1.585    |    0.631
      3      |    2.000    |    0.500
      4      |    2.322    |    0.431
      5      |    2.585    |    0.387
      6      |    2.807    |    0.356
      7      |    3.000    |    0.333
      8      |    3.170    |    0.316
      9      |    3.322    |    0.301
     10      |    3.459    |    0.289   (much less credit)
```

A relevant item at position 1 gets 3.5x more credit than the same item at position 10.

### Step 3: Ideal DCG (IDCG)

The Ideal DCG is the **best possible DCG** -- what you would get if the items were sorted by relevance in descending order. It represents the perfect ranking.

### Step 4: NDCG = DCG / IDCG

Normalizing by the ideal gives a score between 0 and 1:

```
              DCG@K
NDCG@K  =  ─────────
              IDCG@K
```

- NDCG = 1.0 means the ranking is perfect (most relevant items first)
- NDCG = 0.0 means no relevant items at all

### Full worked example

Query movie: **The Dark Knight** (genres: Action, Crime, Drama, Thriller).

10 recommendations with their genre overlaps:

```
Position | Movie                | Overlap count | Discounted contribution
─────────+──────────────────────+───────────────+─────────────────────────
    1    | Dark Knight Rises    |      4        |  4 / log2(2) = 4 / 1.000 = 4.000
    2    | Scarface             |      4        |  4 / log2(3) = 4 / 1.585 = 2.524
    3    | The Prestige         |      2        |  2 / log2(4) = 2 / 2.000 = 1.000
    4    | Training Day         |      4        |  4 / log2(5) = 4 / 2.322 = 1.723
    5    | My Neighbor Totoro   |      0        |  0 / log2(6) = 0 / 2.585 = 0.000
    6    | Heat                 |      4        |  4 / log2(7) = 4 / 2.807 = 1.425
    7    | Running Scared       |      4        |  4 / log2(8) = 4 / 3.000 = 1.333
    8    | Bridget Jones        |      0        |  0 / log2(9) = 0 / 3.170 = 0.000
    9    | The Asphalt Jungle   |      4        |  4 / log2(10)= 4 / 3.322 = 1.204
   10    | Bullitt              |      4        |  4 / log2(11)= 4 / 3.459 = 1.157
```

**DCG@10 = 4.000 + 2.524 + 1.000 + 1.723 + 0.000 + 1.425 + 1.333 + 0.000 + 1.204 + 1.157 = 14.366**

Now compute **IDCG@10** -- sort the relevances in descending order first:

```
Sorted relevances: [4, 4, 4, 4, 4, 4, 4, 2, 0, 0]

Position | Relevance | Discounted contribution
─────────+───────────+─────────────────────────
    1    |     4     | 4 / 1.000 = 4.000
    2    |     4     | 4 / 1.585 = 2.524
    3    |     4     | 4 / 2.000 = 2.000
    4    |     4     | 4 / 2.322 = 1.723
    5    |     4     | 4 / 2.585 = 1.547
    6    |     4     | 4 / 2.807 = 1.425
    7    |     4     | 4 / 3.000 = 1.333
    8    |     2     | 2 / 3.170 = 0.631
    9    |     0     | 0 / 3.322 = 0.000
   10    |     0     | 0 / 3.459 = 0.000

IDCG@10 = 4.000 + 2.524 + 2.000 + 1.723 + 1.547 + 1.425 + 1.333 + 0.631 + 0.000 + 0.000 = 15.183
```

**NDCG@10 = DCG / IDCG = 14.366 / 15.183 = 0.946**

Interpretation: The ranking is 94.6% of the way to ideal. The only "mistake" was placing Totoro (0 overlap) at position 5 and The Prestige (2 overlap) at position 3, when ideally a 4-overlap movie would be there.

### Why NDCG matters more than Precision for ranked lists

Precision treats all positions equally. In reality, position matters enormously:

```
                Precision@10 = 0.80 for both lists, but...

List A:  [relevant, relevant, relevant, relevant, irrelevant, relevant, ...]
          User sees good results immediately -- happy!

List B:  [irrelevant, irrelevant, irrelevant, irrelevant, relevant, relevant, ...]
          User sees bad results first -- leaves the site!

NDCG distinguishes these: List A gets a higher NDCG than List B.
```

### How the code works

```python
def ndcg_at_k(recommender, test_movies, k=10):
    ndcg_scores = []

    for _, row in test_movies.iterrows():
        title = row['title']
        true_genres = row['genres']

        recs = recommender.recommend(title, n=k)

        # Compute relevance for each recommendation (genre overlap count)
        relevances = []
        for _, rec_row in recs.iterrows():
            rec_genres = rec_row['genres']
            relevances.append(_genre_overlap(true_genres, rec_genres))

        # The "ideal" is the same relevances sorted descending
        true_relevance = np.array([sorted(relevances, reverse=True)])
        pred_relevance = np.array([relevances])
        ndcg_scores.append(ndcg_score(true_relevance, pred_relevance, k=k))

    return np.mean(ndcg_scores)
```

The code uses scikit-learn's `ndcg_score`, which takes two arrays:

- `true_relevance`: the ideal ordering (sorted descending)
- `pred_relevance`: the actual ordering from our system

It computes NDCG automatically and handles all the logarithmic discounting.

---

## 4. Coverage

### What it measures

Coverage answers: **What fraction of the total movie catalog ever appears in any recommendation list?**

### The formula

```
                    |unique recommended movies across all test queries|
Coverage  =  ──────────────────────────────────────────────────────────
                              |total movies in catalog|
```

### Why it matters

Consider a catalog of 9,219 movies. If the system only ever recommends the same 200 popular blockbusters regardless of the query, that is:

```
Coverage = 200 / 9219 = 0.022  (2.2%)
```

This means 97.8% of the catalog is invisible to users. The long tail of lesser-known but potentially great movies never gets recommended.

### The popularity bias problem

Recommendation systems have a natural tendency toward **popularity bias**:

```
                |
   # of times   |  ****
   recommended  |  *   ****
                |  *       ****
                |  *           *********
                |  *                    ***********************
                +──────────────────────────────────────────────
                   Movies sorted by popularity (most to least)

                   ^^^^^^^                ^^^^^^^^^^^^^^^^^^^^^^^^
                   These few popular      These thousands of movies
                   movies dominate        never get recommended
                   all lists
```

Why does this happen?

- Popular movies have more data (reviews, ratings, metadata), so the system learns more about them.
- Popular movies appear in training data more often.
- Systems that include popularity signals (like weighted ratings) naturally favor well-known films.

Coverage quantifies this problem. Low coverage = high popularity bias.

### How the code works

```python
def coverage(recommender, test_movies, catalog_size, k=10):
    all_rec_titles = set()    # collect ALL unique movie titles ever recommended

    for _, row in test_movies.iterrows():
        recs = recommender.recommend(row['title'], n=k)
        if not recs.empty:
            all_rec_titles.update(recs['title'].tolist())

    return len(all_rec_titles) / catalog_size
```

The function loops over all test movies, collects every unique title that appears in any recommendation list (using a set to avoid duplicates), and divides by the total catalog size.

### Example calculation

If we test with 100 query movies, each getting 10 recommendations, we produce 1,000 total recommendation slots. But many slots will contain the same popular movies. Suppose only 1,389 unique movies ever appear:

```
Coverage = 1389 / 9219 = 0.151  (15.1%)
```

This means 84.9% of the catalog never gets recommended across all test queries. While this might seem low, it is typical for content-based systems that match on specific feature similarity. Not every movie is similar to the test set.

---

## 5. Intra-List Diversity (ILD)

### What it measures

ILD measures **how different the items within a single recommendation list are from each other**. It answers: if you look at one recommendation list, are all the movies basically the same, or is there variety?

### The problem ILD detects

Even if all recommendations are "relevant," a list of 10 nearly identical movies is not useful:

```
BAD (low diversity):                    GOOD (high diversity):
 1. Avengers: Endgame                   1. The Dark Knight Rises
 2. Avengers: Infinity War              2. Scarface
 3. Avengers: Age of Ultron             3. The Prestige
 4. Captain America: Civil War          4. Training Day
 5. Iron Man 3                          5. Heat
 6. Thor: Ragnarok                      6. Harry Brown
 7. Guardians of the Galaxy             7. Running Scared
 8. Black Panther                       8. The Asphalt Jungle
 9. Doctor Strange                      9. Bullitt
10. Ant-Man                            10. Thursday

All Marvel superhero films --           Mix of crime films across
user probably knows them all.           different decades and styles.
ILD is low.                             ILD is high.
```

### How it is computed

ILD uses the **cosine similarity matrix** that the recommender already computes during fitting. This matrix tells us how similar any two movies are based on their full feature representation (overview, genres, keywords, cast, director, etc.).

For each recommendation list:

1. Take all pairs of recommended movies
2. Look up their cosine similarity from the precomputed matrix
3. Compute the mean pairwise similarity
4. Subtract from 1 (so higher = more diverse)

```
ILD = 1 - mean(pairwise cosine similarity within the recommendation list)
```

### Why "1 minus similarity"?

Cosine similarity goes from 0 (completely different) to 1 (identical):

- If all pairs have similarity 1.0 (identical movies), ILD = 1 - 1.0 = 0.0 (no diversity)
- If all pairs have similarity 0.0 (totally different), ILD = 1 - 0.0 = 1.0 (maximum diversity)

### Worked example

Suppose we recommend 4 movies (using 4 for simplicity; real system uses 10), and the pairwise cosine similarities between them are:

```
              Movie A   Movie B   Movie C   Movie D
Movie A         1.0       0.8       0.3       0.2
Movie B         0.8       1.0       0.4       0.1
Movie C         0.3       0.4       1.0       0.6
Movie D         0.2       0.1       0.6       1.0
```

We only need the upper triangle (each pair counted once, excluding diagonal):

```
Pairs:
  A-B: 0.8
  A-C: 0.3
  A-D: 0.2
  B-C: 0.4
  B-D: 0.1
  C-D: 0.6
```

Mean pairwise similarity = (0.8 + 0.3 + 0.2 + 0.4 + 0.1 + 0.6) / 6 = 2.4 / 6 = 0.40

**ILD = 1 - 0.40 = 0.60**

Interpretation: The list has moderate diversity. Movies A and B are quite similar (0.8), but the other pairs are more varied.

### How the code works

```python
def intra_list_diversity(recommender, test_movies, k=10):
    diversities = []

    for _, row in test_movies.iterrows():
        title = row['title']
        recs = recommender.recommend(title, n=k)

        # Find indices of recommended movies in the cosine similarity matrix
        rec_indices = []
        for t in recs['title'].tolist():
            if t in recommender.indices:
                rec_indices.append(recommender.indices[t])

        # Extract the submatrix for just these movies
        sub_sim = recommender.cosine_sim[np.ix_(rec_indices, rec_indices)]

        # Mean of upper triangle (pairs without self-similarity)
        n_items = len(rec_indices)
        upper_mask = np.triu_indices(n_items, k=1)   # k=1 skips diagonal
        mean_sim = sub_sim[upper_mask].mean()
        diversities.append(1.0 - mean_sim)

    return np.mean(diversities)
```

Key details:

- `np.triu_indices(n, k=1)` gets indices above the main diagonal, so each pair is counted once and no movie is compared with itself.
- `np.ix_` extracts a submatrix from the full cosine similarity matrix.
- The final result is averaged across all test queries.

---

## 6. Novelty

### What it measures

Novelty quantifies how "surprising" or "unexpected" the recommendations are. It answers: **is the system recommending well-known blockbusters that everyone has heard of, or is it surfacing hidden gems?**

### The intuition from information theory

Novelty is based on **self-information** from information theory. The core idea:

> An event that is very common carries little information.
> An event that is rare carries a lot of information.

Applied to recommendations:

- Recommending a movie that everyone watches (high popularity) is **not surprising** = low novelty.
- Recommending a movie that few people have watched (low popularity) **is surprising** = high novelty.

### The formula

For a single recommended movie:

```
novelty = -log2(normalized_popularity)
```

Where:

- `normalized_popularity` = the movie's popularity score divided by the maximum popularity in the catalog, giving a value in (0, 1].
- `log2` is the base-2 logarithm.
- The negative sign makes the result positive (since log of a number less than 1 is negative).

The overall novelty score is the **mean** of this value across all recommended items across all test queries.

### Why logarithm?

The logarithm matches our intuitive sense of surprise:

```
Normalized     |  -log2(popularity)  |  Interpretation
popularity     |  = novelty          |
───────────────+─────────────────────+──────────────────────────
  1.00         |     0.00            |  The most popular movie.
               |                     |  Zero surprise.
  0.50         |     1.00            |  Half as popular.
               |                     |  Mildly surprising.
  0.10         |     3.32            |  10x less popular.
               |                     |  Quite surprising.
  0.01         |     6.64            |  100x less popular.
               |                     |  Very surprising.
  0.001        |     9.97            |  1000x less popular.
               |                     |  Extremely novel discovery.
```

The log scale means that going from "very popular" to "somewhat popular" does not change novelty much, but going from "somewhat popular" to "obscure" increases novelty dramatically. This matches intuition: recommending the 5th most popular movie instead of the 1st is barely different, but recommending a film ranked 5000th is genuinely novel.

### The self-information interpretation

In information theory, self-information measures "how much information" an event carries. If you model movie watching as a random process where a movie's probability of being watched is proportional to its popularity, then:

```
self-information = -log2(P(movie being watched))
```

This is measured in **bits**. A novelty of 6.0 means the recommendation carries about 6 bits of information -- it tells the user something they were very unlikely to know about.

### Worked example

Suppose our catalog has a maximum popularity of 500. We recommend 3 movies:

```
Movie              | Popularity | Normalized     | Novelty = -log2(norm)
───────────────────+────────────+────────────────+──────────────────────
Avengers: Endgame  |    500     | 500/500 = 1.00 | -log2(1.00) = 0.00
Inception          |    100     | 100/500 = 0.20 | -log2(0.20) = 2.32
Thursday           |      5     |   5/500 = 0.01 | -log2(0.01) = 6.64
```

Mean novelty = (0.00 + 2.32 + 6.64) / 3 = **2.99**

If we replaced Avengers: Endgame with another obscure film:

```
Harry Brown        |      8     |   8/500 = 0.016 | -log2(0.016) = 5.97
Inception          |    100     | 100/500 = 0.20  | -log2(0.20)  = 2.32
Thursday           |      5     |   5/500 = 0.01  | -log2(0.01)  = 6.64
```

Mean novelty = (5.97 + 2.32 + 6.64) / 3 = **4.98** (much higher!)

### How the code works

```python
def novelty(recommender, test_movies, popularity_scores, k=10):
    pop_max = popularity_scores.max()

    # Normalize popularity to [0, 1]
    pop_norm = popularity_scores / pop_max
    pop_by_title = pd.Series(pop_norm.values, index=recommender.smd['title'])

    novelty_scores = []

    for _, row in test_movies.iterrows():
        recs = recommender.recommend(row['title'], n=k)

        for _, rec_row in recs.iterrows():
            p = float(pop_by_title.get(rec_row['title'], 0.5))  # default 0.5 if unknown
            novelty_scores.append(-np.log2(max(p, 1e-10)))       # clamp to avoid log(0)

    return np.mean(novelty_scores)
```

Key details:

- `max(p, 1e-10)` prevents `log2(0)` which would be negative infinity.
- Unknown movies default to popularity 0.5 (medium novelty).
- Individual novelty scores are collected per recommended item (not per query), then averaged.

---

## 7. Serendipity@K

### What it measures

Serendipity captures both **surprise and relevance** simultaneously. A novel but irrelevant recommendation is not serendipitous -- it is just random. A relevant but expected recommendation is useful but not serendipitous. Serendipity rewards recommendations that are **unexpectedly good**.

### The formula

For each recommended movie:

```
serendipity = unexpectedness x relevance
```

Where:

- `unexpectedness = 1 - normalized_popularity` (less popular = more unexpected)
- `relevance = 1` if the movie shares at least one genre with the query, else `0`

The overall Serendipity@K is the mean across all recommendations across all test queries.

### Intuition

```
                  Relevant    Not Relevant
Unexpected        Serendipitous!   Just noise
Expected          Useful           Useless
```

A popular, relevant movie (e.g., recommending Avengers to a Marvel fan) gets low serendipity because `unexpectedness` is near zero. An obscure but relevant movie (a little-known crime thriller recommended to someone who liked The Dark Knight) scores high on both factors.

### Worked example

Query movie: The Dark Knight (genres: Action, Crime, Drama, Thriller). Max popularity in catalog = 500.

```
Movie                  | Popularity | Unexpectedness     | Genre overlap? | Serendipity
-----------------------+------------+--------------------+----------------+------------
Avengers: Endgame      |    500     | 1 - 500/500 = 0.00| Yes            | 0.00 x 1 = 0.00
Training Day           |     30     | 1 - 30/500 = 0.94 | Yes            | 0.94 x 1 = 0.94
My Neighbor Totoro     |     40     | 1 - 40/500 = 0.92 | No             | 0.92 x 0 = 0.00
Thursday               |      5     | 1 - 5/500 = 0.99  | Yes            | 0.99 x 1 = 0.99
```

Mean serendipity = (0.00 + 0.94 + 0.00 + 0.99) / 4 = **0.48**

### How the code works

```python
def serendipity_at_k(recommender, test_movies, popularity_scores, k=10):
    for _, row in test_movies.iterrows():
        recs = recommender.recommend(title, n=k)
        for _, rec_row in recs.iterrows():
            relevance = 1.0 if _genre_overlap(true_genres, rec_genres) > 0 else 0.0
            p = float(pop_by_title.get(rec_row['title'], 0.5))
            unexpectedness = 1.0 - p
            scores.append(unexpectedness * relevance)
    return np.mean(scores)
```

### How serendipity differs from novelty

- **Novelty** measures surprise regardless of relevance (log-based, all recommendations count)
- **Serendipity** requires both surprise AND relevance (multiplicative, irrelevant items score 0)

A system with high novelty but low serendipity recommends many obscure movies that are not relevant. A system with high serendipity finds hidden gems the user would actually enjoy.

---

## 8. Mean Reciprocal Rank (MRR)

### What it measures

MRR answers: **How far down the list does the user have to scroll before finding the first relevant recommendation?**

Unlike Precision@K (which counts all relevant items) or NDCG (which cares about the full ranking), MRR focuses exclusively on the **position of the first hit**.

### The formula

For a single query:

```
                    1
Reciprocal Rank = ─────
                   rank

where rank = position of the first relevant item (1-indexed)
```

MRR is the mean of reciprocal ranks across all test queries.

### Worked example

```
Query 1: First relevant item at position 1  →  RR = 1/1 = 1.000
Query 2: First relevant item at position 3  →  RR = 1/3 = 0.333
Query 3: First relevant item at position 1  →  RR = 1/1 = 1.000
Query 4: No relevant items in top 10        →  RR = 0.000

MRR = (1.000 + 0.333 + 1.000 + 0.000) / 4 = 0.583
```

### Why MRR matters

MRR captures the user's **first impression**. Even if Precision@10 is 0.90, if the one irrelevant item happens to be at position 1, the user's first experience is bad. MRR penalizes this heavily.

### How the code works

```python
def mean_reciprocal_rank(recommender, test_movies, k=10):
    for _, row in test_movies.iterrows():
        recs = recommender.recommend(title, n=k)
        for rank, (_, rec_row) in enumerate(recs.iterrows(), 1):
            if _genre_overlap(true_genres, rec_genres) > 0:
                rr_scores.append(1.0 / rank)
                break
        else:
            rr_scores.append(0.0)   # no relevant item found
    return np.mean(rr_scores)
```

The `for...else` pattern: the `else` block runs only if the loop completes without `break` — meaning no relevant item was found in the top K.

---

## 9. Per-Genre Precision

### What it measures

Per-Genre Precision breaks down Precision@K by individual genre. Instead of one aggregate number, it shows how well the system performs for each genre category.

### Why it matters

A system with Precision@10 = 0.90 overall might be excellent for Action movies (0.98) but poor for Documentary (0.60). Per-genre breakdown reveals these disparities. This is important because:

- Users who like niche genres deserve good recommendations too
- It highlights which genres the system struggles with (possibly due to sparse data)
- It guides targeted improvements (e.g., increasing keyword weight might help documentaries)

### How it works

For each test movie, the function computes Precision@K as usual. It then attributes that precision score to **every genre** the test movie belongs to. Genres with fewer than 3 test movies are excluded to avoid noisy estimates.

```python
def per_genre_precision(recommender, test_movies, k=10):
    genre_results = {}
    for _, row in test_movies.iterrows():
        # Compute precision for this query
        prec = relevant / k
        # Attribute to each of this movie's genres
        for g in true_genres:
            genre_results.setdefault(g, []).append(prec)
    # Average per genre, require at least 3 samples
    return {g: np.mean(v) for g, v in genre_results.items() if len(v) >= 3}
```

### Example output

```
Genre          | Precision@10 | # Test Movies
---------------+--------------+--------------
Action         |     0.9850   |     42
Drama          |     0.9920   |     58
Comedy         |     0.9780   |     35
Horror         |     0.9500   |     12
Documentary    |     0.8200   |      5
Animation      |     0.9600   |      8
```

---

## 10. Grid Search for Weight Optimization

### What problem does it solve?

Our recommendation system combines multiple fields (genres, keywords, director, cast, etc.) into a single feature representation. Each field gets a **weight** that controls how much it influences the final similarity score. But how do we pick the best weights?

The default weights in our system are:

```python
DEFAULT_WEIGHTS = {
    'overview':   1.0,   # Movie plot description
    'genres':     1.5,   # Genre labels (Action, Comedy, etc.)
    'keywords':   1.2,   # Plot keywords
    'cast':       1.0,   # Actors
    'director':   2.0,   # Director
    'decade':     0.3,   # Release decade
    'language':   0.5,   # Original language
    'collection': 1.5,   # Franchise/collection
}
```

These were chosen by intuition. But we can do better -- we can **systematically try many combinations** and see which performs best. This is called **grid search**.

### Which weights are varied?

The grid search varies four fields (keeping the others at their default values):

```
Field      | Values tried
───────────+──────────────────────
genres     | 0.5,  1.0,  1.5,  2.0
keywords   | 0.5,  1.0,  1.5,  2.0
director   | 1.0,  1.5,  2.0,  2.5
collection | 0.5,  1.0,  1.5,  2.0
```

### How many combinations?

Each field has 4 possible values, and there are 4 fields:

```
Total combinations = 4 x 4 x 4 x 4 = 256
```

For each of these 256 combinations, the system:

1. Creates a new recommender with those weights
2. Fits the recommender (computes TF-IDF matrices and cosine similarity)
3. Evaluates Precision@K and NDCG@K on the test set
4. Records the results

### The objective function

The grid search optimizes a **combined score**:

```
combined_score = 0.5 * Precision@K + 0.5 * NDCG@K
```

Equal weight is given to both metrics, meaning we want recommendations that are both relevant (Precision) and well-ranked (NDCG).

### How it works step by step

```
Step 1: Generate all 256 weight combinations
        ─────────────────────────────────────
        combo #1:   genres=0.5, keywords=0.5, director=1.0, collection=0.5
        combo #2:   genres=0.5, keywords=0.5, director=1.0, collection=1.0
        combo #3:   genres=0.5, keywords=0.5, director=1.0, collection=1.5
        ...
        combo #256: genres=2.0, keywords=2.0, director=2.5, collection=2.0

Step 2: For each combo, build recommender, evaluate, record results
        ──────────────────────────────────────────────────────────────
        combo #1  → P@10=0.72, NDCG=0.80, combined=0.76
        combo #2  → P@10=0.74, NDCG=0.82, combined=0.78
        ...
        combo #137 → P@10=0.99, NDCG=0.97, combined=0.98  ← best!
        ...
        combo #256 → P@10=0.91, NDCG=0.90, combined=0.905

Step 3: Sort by combined score, return best weights
        ─────────────────────────────────────────────
        Best: genres=1.5, keywords=1.0, director=2.0, collection=1.5
        Score: 0.98
```

### How the code works

```python
def grid_search_weights(df, test_movies, weight_ranges=None, k=10):
    if weight_ranges is None:
        weight_ranges = {
            'genres':     [0.5, 1.0, 1.5, 2.0],
            'keywords':   [0.5, 1.0, 1.5, 2.0],
            'director':   [1.0, 1.5, 2.0, 2.5],
            'collection': [0.5, 1.0, 1.5, 2.0],
        }

    # itertools.product generates all combinations
    combos = list(product(*value_lists))   # 256 combinations

    for combo in combos:
        weights = DEFAULT_WEIGHTS.copy()   # start from defaults
        for field, val in zip(fields, combo):
            weights[field] = val           # override the 4 search fields

        rec = ContentBasedRecommender(df, weights=weights)
        rec.fit()                          # rebuild with new weights

        p_at_k = precision_at_k(rec, test_movies, k)
        n_at_k = ndcg_at_k(rec, test_movies, k)
        combined = 0.5 * p_at_k + 0.5 * n_at_k   # objective function

        if combined > best_score:
            best_score = combined
            best_weights = weights.copy()

    return {'best_weights': best_weights, 'best_score': best_score, 'results': ...}
```

### How to interpret results

The grid search returns:

- `best_weights`: the weight combination that maximized the combined score
- `best_score`: the achieved combined score
- `results`: all 256 combinations sorted by score (useful for analysis)

Things to look for:

- **Dominant fields**: if the best weights put high values on genres and director but low on keywords, it means genre and director similarity are more important for relevance.
- **Diminishing returns**: if increasing a weight from 1.5 to 2.0 barely changes the score, the field is already well-represented.
- **Sensitivity**: if small weight changes cause large score changes, the system is sensitive to that field's weight.

### Limitations of grid search

- **Computational cost**: 256 combinations, each requiring a full fit + evaluate cycle. At ~0.4s per fit + evaluation time, this can take several minutes.
- **Coarse granularity**: testing only 4 values per field might miss the true optimum (e.g., genres=1.3 might be better than 1.0 or 1.5, but we never test it).
- **Overfitting**: optimizing on the test set means the "best" weights might not generalize to new data. Ideally, you would use a separate validation set.

---

## 11. The Evaluation Procedure

### The `if __name__ == '__main__'` block

At the bottom of `evaluate_model.py` is the main evaluation script:

```python
if __name__ == '__main__':
    # Step 1: Load and process data
    smd = make_dataset()         # Load 5 CSVs, clean, merge → 9,219 movies
    smd = build_features(smd)    # Create text features (soup, cleaned columns)

    # Step 2: Build and fit the recommender
    rec = ContentBasedRecommender(smd)
    rec.fit()                    # ~0.4 seconds

    # Step 3: Sample test movies
    test = smd[smd['vote_count'] >= smd['vote_count'].quantile(0.6)].sample(
        n=min(100, len(smd)), random_state=42
    )

    # Step 4: Run all metrics
    results = evaluate_all(rec, test, k=10)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
```

### Step 3 in detail: Sampling test movies

The test set sampling has three important choices:

**1. Filter: `vote_count >= 60th percentile`**

Only movies with enough votes are included as test queries. Why?

- Movies with very few votes have unreliable metadata.
- We want query movies that have sufficient data for meaningful recommendations.
- The 60th percentile means we use the top 40% of movies by vote count.

**2. Sample size: `n = min(100, len(smd))`**

We randomly sample 100 movies from the filtered set. Why 100?

- Enough to get stable averages (reduce variance in metric estimates).
- Small enough to run in reasonable time (each query generates recommendations, and we compute pairwise similarities for ILD).

**3. Random seed: `random_state=42`**

Using a fixed seed makes results **reproducible**. Anyone running the same code on the same data gets exactly the same 100 test movies and the same metric values.

### Step 4 in detail: What `evaluate_all` runs

```python
def evaluate_all(recommender, test_movies, k=10):
    catalog_size = len(recommender.smd)
    pop_scores = pd.to_numeric(recommender.smd['popularity'], errors='coerce').fillna(0.0)

    return {
        'precision_at_k':        precision_at_k(recommender, test_movies, k),
        'ndcg_at_k':             ndcg_at_k(recommender, test_movies, k),
        'coverage':              coverage(recommender, test_movies, catalog_size, k),
        'intra_list_diversity':  intra_list_diversity(recommender, test_movies, k),
        'novelty':               novelty(recommender, test_movies, pop_scores, k),
        'serendipity_at_k':      serendipity_at_k(recommender, test_movies, pop_scores, k),
        'mrr':                   mean_reciprocal_rank(recommender, test_movies, k),
        'per_genre_precision':   per_genre_precision(recommender, test_movies, k),
    }
```

All eight metrics are computed on the same test set, so results are directly comparable.

### Interpreting the output

The output looks like:

```
=== Evaluation Results ===
  precision_at_k: 0.9920
  ndcg_at_k: 0.9650
  coverage: 0.1510
  intra_list_diversity: 0.7290
  novelty: 6.0990
  serendipity_at_k: 0.6340
  mrr: 0.9850
  per_genre_precision: {Action: 0.985, Drama: 0.992, ...}
```

How to read each number:

| Metric                 | Value                                                 | Meaning                                            |
| ---------------------- | ----------------------------------------------------- | -------------------------------------------------- |
| Precision@10 = 0.992   | 99.2% of recommendations share a genre with the query | Near-perfect genre relevance                       |
| NDCG@10 = 0.965        | Ranking is 96.5% of ideal                             | High-overlap items appear near the top             |
| Coverage = 0.151       | 15.1% of catalog gets recommended                     | Moderate; most movies are too dissimilar to appear |
| ILD = 0.729            | Items within lists are moderately diverse             | Not all identical, but genre-coherent              |
| Novelty = 6.099        | Average ~6 bits of self-information                   | Recommends fairly obscure movies                   |
| Serendipity@10 = 0.634 | Combines unexpectedness with relevance                | Finds surprising relevant movies                   |
| MRR = 0.985            | First relevant item is almost always at position 1    | Excellent first-hit placement                      |
| Per-Genre Precision    | Breakdown by genre                                    | Shows genre-level performance variation            |

---

## 12. Actual Results from Our System

The evaluation report compares two versions of the recommender: V3 (single soup-based TF-IDF) and V4 (per-field vectorizers with explicit weights):

**Test setup**: 9,219 movies, 200 test movies with vote_count >= 60th percentile, K = 10.

```
Metric                 | V3 (Baseline) | V4 (New)  | Change
───────────────────────+───────────────+───────────+────────────────
Precision@10           |     0.743     |   0.992   | +0.249 (+33.6%)
NDCG@10                |     0.821     |   0.965   | +0.144 (+17.5%)
Coverage               |     0.142     |   0.151   | +0.010 (+6.8%)
Intra-List Diversity   |     0.935     |   0.729   | -0.206 (-22.0%)
Novelty                |     5.682     |   6.099   | +0.417 (+7.3%)
```

### What the numbers tell us

**Precision@10 jumped from 0.743 to 0.992.** This is the most dramatic improvement. V3's single "soup" approach mixed all text features into one bag of words, which diluted genre signals. V4's per-field vectorizers with CountVectorizer for genres (instead of TF-IDF, which would penalize common genre names) produce far more genre-coherent recommendations.

**NDCG@10 improved from 0.821 to 0.965.** Not only are more recommendations relevant, but the most relevant ones (highest genre overlap) are ranked higher. The combined scoring formula (`0.7 * similarity + 0.3 * weighted_rating`) plus MMR re-ranking produces well-ordered lists.

**Coverage barely changed (0.142 to 0.151).** Both systems recommend about 15% of the catalog. This is a known limitation of content-based filtering -- it can only recommend movies that are similar to the query, and many movies in the catalog are simply not similar enough to any test query.

**ILD dropped from 0.935 to 0.729.** This is the one metric where V4 is "worse," but it is an **intentional tradeoff**:

```
V3: "soup" mixes all signals → noisier similarity → recommendations are
     more varied (high ILD) but less relevant (lower Precision)

V4: per-field weights give genres strong influence → recommendations
     share genres more consistently → less varied within lists (lower ILD)
     but far more relevant (higher Precision)
```

An ILD of 0.729 is still solid. The MMR (Maximal Marginal Relevance) re-ranking with lambda=0.5 prevents the list from degenerating into 10 identical movies. Without MMR, ILD would be even lower.

**Novelty increased from 5.682 to 6.099.** V4 recommends slightly more obscure movies. Why? V3 used a hard vote_count cutoff (only movies above the 60th percentile of vote_count could be recommended), which eliminated most lesser-known films. V4 uses a soft scoring approach (`0.7 * similarity + 0.3 * normalized_weighted_rating`) that does not exclude low-vote movies entirely, allowing more niche films to surface.

### The tradeoff diagram for our system

```
Metric        0.0                          0.5                           1.0
              |                             |                             |
Precision     |                             |                         V3==|===V4
              |                             |                             |
NDCG          |                             |                      V3=====|=V4
              |                             |                             |
Coverage      |  V3=V4                      |                             |
              |                             |                             |
ILD           |                             |              V4=======V3    |
              |                             |                             |
Novelty*      |                             |      V3========V4           |
              |                             |                             |

* Novelty is on a different scale (bits); shown here qualitatively.
```

V4 wins decisively on accuracy metrics (Precision, NDCG) while maintaining competitive coverage and novelty. The ILD decrease is the cost of better genre coherence -- a worthwhile tradeoff since the recommendations are now far more relevant.

### Example: "The Dark Knight" recommendations

The evaluation report includes a concrete comparison:

**V3** recommended 6 out of 10 Batman-related movies (franchise echo chamber) and included The Lego Movie (similarity 0.078) because the TF-IDF boosted the rare "Batman" token. The list was sorted by weighted_rating rather than similarity, so The Dark Knight Rises (highest similarity at 0.501) appeared at position 5 instead of position 1.

**V4** recommended only 1 Batman film (the direct sequel) and filled the rest with diverse crime thrillers: Scarface, The Prestige, Training Day, Heat, Bullitt, The Asphalt Jungle. CountVectorizer for genres avoided the IDF penalty that made V3 over-index on character names. MMR ensured variety.

---

## Summary of All Metrics

```
+──────────────────────+────────────────────────────────+───────────+──────────────+
| Metric               | What it measures               | Range     | Good value   |
+──────────────────────+────────────────────────────────+───────────+──────────────+
| Precision@K          | Fraction of recs that are      | [0, 1]    | > 0.8        |
|                      | relevant (genre overlap)       |           |              |
+──────────────────────+────────────────────────────────+───────────+──────────────+
| NDCG@K               | Are the BEST items ranked      | [0, 1]    | > 0.8        |
|                      | highest?                       |           |              |
+──────────────────────+────────────────────────────────+───────────+──────────────+
| MRR                  | How quickly does the first     | [0, 1]    | > 0.9        |
|                      | relevant item appear?          |           |              |
+──────────────────────+────────────────────────────────+───────────+──────────────+
| Coverage             | Fraction of catalog ever       | [0, 1]    | Higher is    |
|                      | recommended                    |           | better       |
+──────────────────────+────────────────────────────────+───────────+──────────────+
| Intra-List Diversity | How different are items        | [0, 1]    | 0.5 - 0.9    |
|                      | within one list?               |           |              |
+──────────────────────+────────────────────────────────+───────────+──────────────+
| Novelty              | How surprising/obscure are     | [0, inf)  | Higher is    |
|                      | the recommendations? (bits)    |           | more novel   |
+──────────────────────+────────────────────────────────+───────────+──────────────+
| Serendipity@K        | Unexpected AND relevant        | [0, 1]    | Higher is    |
|                      | recommendations?               |           | better       |
+──────────────────────+────────────────────────────────+───────────+──────────────+
| Per-Genre Precision  | Precision broken down by       | [0, 1]    | Even across  |
|                      | individual genre category      | per genre | genres       |
+──────────────────────+────────────────────────────────+───────────+──────────────+
```

No single metric captures "good recommendations." You must look at all of them together and understand the tradeoffs your system makes.
