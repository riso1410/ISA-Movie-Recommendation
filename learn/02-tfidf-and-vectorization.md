# TF-IDF and Text Vectorization

How text descriptions of movies become numbers that a computer can use to find similar films.

---

## Table of Contents

1. [The Fundamental Problem](#1-the-fundamental-problem)
2. [Bag of Words](#2-bag-of-words)
3. [TF-IDF (Term Frequency -- Inverse Document Frequency)](#3-tf-idf-term-frequency--inverse-document-frequency)
4. [CountVectorizer vs TfidfVectorizer](#4-countvectorizer-vs-tfidfvectorizer)
5. [The Multi-Vectorizer Architecture](#5-the-multi-vectorizer-architecture)
6. [Sparse Matrices](#6-sparse-matrices)

---

## 1. The Fundamental Problem

We have text descriptions of movies. For example:

| Movie           | Overview                                  |
| --------------- | ----------------------------------------- |
| The Dark Knight | "Batman fights the Joker in Gotham City"  |
| Iron Man        | "Tony Stark builds a powered armor suit"  |
| The Notebook    | "A young couple falls in love one summer" |

We want a computer to answer the question: **"Which two of these movies are most similar?"**

A human can read these descriptions and reason about them. But a computer cannot do
math on words. It cannot subtract "Batman" from "Iron Man" or calculate the distance
between "love" and "armor." Computers operate on numbers -- specifically, on vectors
(ordered lists of numbers).

So the core challenge is:

```
TEXT  --->  ???  --->  NUMBERS
```

We need a translation step. We need to convert each movie's text into a list of
numbers (a vector) such that movies with similar text end up with similar vectors.

Once we have numbers, we can use mathematical tools like cosine similarity
(covered in the next learning document) to measure how close two movies are.

**The full pipeline looks like this:**

```
"Batman fights the Joker"  --->  [0.0, 0.41, 0.0, 0.58, 0.0, ...]  --->  similarity = 0.83
"A hero battles villains"  --->  [0.0, 0.0, 0.52, 0.0, 0.47, ...]  --->  with another movie
```

The question is: how do we get from the text on the left to the numbers on the right?

---

## 2. Bag of Words

Before we tackle TF-IDF, we need to understand the simplest text-to-numbers method:
**Bag of Words (BoW).**

### The Idea

Imagine you take a movie description, rip out every word, throw them all into a bag,
and shake it up. You lose all word order -- you only know _which_ words appear and
_how many times_ each one appears. That is Bag of Words.

### Step-by-Step Example

Let us work with three tiny movie "descriptions":

```
Movie A: "action hero saves city"
Movie B: "hero fights villain city"
Movie C: "romantic love story drama"
```

#### Step 1: Build a Vocabulary

Collect every unique word across all documents, and sort them alphabetically. This
becomes our vocabulary:

```
Vocabulary: [action, city, drama, fights, hero, love, romantic, saves, story, villain]
Position:   [  0,     1,    2,     3,      4,    5,     6,       7,     8,      9   ]
```

There are 10 unique words, so each movie will be represented by a vector of 10 numbers.

#### Step 2: Count Words

For each movie, count how many times each vocabulary word appears:

```
             action  city  drama  fights  hero  love  romantic  saves  story  villain
Movie A:   [  1,      1,    0,     0,      1,    0,     0,       1,     0,      0   ]
Movie B:   [  0,      1,    0,     1,      1,    0,     0,       0,     0,      1   ]
Movie C:   [  0,      0,    1,     0,      0,    1,     1,       0,     1,      0   ]
```

Now every movie is a vector of numbers. Movie A is `[1, 1, 0, 0, 1, 0, 0, 1, 0, 0]`.

#### Step 3: Observe

- Movie A and Movie B both have "city" (position 1) and "hero" (position 4) as 1.
  They share vocabulary. Their vectors partially overlap.
- Movie C shares zero words with A or B. Its vector has 1s in completely different
  positions.
- Intuitively: A and B are more similar. The numbers reflect this.

#### The Matrix View

We can stack these vectors into a matrix:

```
              action  city  drama  fights  hero  love  romantic  saves  story  villain
            ┌──────────────────────────────────────────────────────────────────────────┐
  Movie A   │   1      1      0      0      1     0      0        1      0       0    │
  Movie B   │   0      1      0      1      1     0      0        0      0       1    │
  Movie C   │   0      0      1      0      0     1      1        0      1       0    │
            └──────────────────────────────────────────────────────────────────────────┘

  Shape: 3 movies x 10 words = (3, 10) matrix
```

Each **row** is one movie. Each **column** is one word from the vocabulary.

### Limitations of Bag of Words

Bag of Words has a serious flaw. Consider a real movie overview:

> "The hero goes to the city and the people of the city welcome the hero."

The word "the" appears 4 times. The word "hero" appears 2 times. In a raw word-count
vector, "the" would have a value of 4 and "hero" a value of 2. But "the" tells us
absolutely nothing about what the movie is about. It is just a common English word
that appears in every single document.

**The problem:** common words dominate the vector, drowning out the rare, meaningful
words. If "the" appears in every movie description, it cannot help us distinguish
one movie from another.

This is exactly the problem TF-IDF was invented to solve.

---

## 3. TF-IDF (Term Frequency -- Inverse Document Frequency)

TF-IDF is a way to assign a **weight** to every word in every document. The weight
is high when a word is **frequent in this document** but **rare across all documents**.
The weight is low when a word is common everywhere.

The name itself tells you the two parts:

- **TF** = Term Frequency (how often does this word appear _here_?)
- **IDF** = Inverse Document Frequency (how rare is this word _everywhere_?)

Let us break down each part.

---

### 3.1 Term Frequency (TF)

**What it measures:** How often a word appears in a single document, relative to the
document length.

**Raw formula:**

```
                    count of word t in document d
TF(t, d) = ──────────────────────────────────────────
             total number of words in document d
```

**Worked example:**

Document: "action hero saves the city hero" (6 words total)

```
TF("hero", d)   = 2/6 = 0.333
TF("action", d) = 1/6 = 0.167
TF("city", d)   = 1/6 = 0.167
TF("the", d)    = 1/6 = 0.167
TF("saves", d)  = 1/6 = 0.167
```

"hero" appears twice, so it gets the highest TF. Makes sense -- this document
emphasizes "hero" more than the other words.

> **Note:** scikit-learn's TfidfVectorizer uses raw counts by default (not divided by
> document length), because the L2 normalization applied at the end achieves a similar
> effect. For conceptual understanding, the ratio form is clearer.

---

### 3.2 Inverse Document Frequency (IDF)

**What it measures:** How rare or common a word is across the _entire collection_ of
documents (called the "corpus").

**Why we need it:** Term Frequency alone cannot tell us if a word is _informative_.
The word "the" might appear 10 times in a document (high TF), but it also appears in
every single document in the corpus. It carries zero information for distinguishing
documents. We need a way to penalize such ubiquitous words and reward rare ones.

**The intuition:**

- A word that appears in every document (like "the") --> IDF is low (near zero)
- A word that appears in only one document (like "gotham") --> IDF is high
- IDF says: "How _special_ is this word?"

**Formula:**

```
                       N
IDF(t) = log ────────────────
              1 + df(t)
```

Where:

- `N` = total number of documents in the corpus
- `df(t)` = number of documents that contain word `t`
- `log` = natural logarithm (base e)
- The `1 +` in the denominator prevents division by zero

**Worked example:**

Corpus of 5 movie descriptions:

```
Doc 1: "action hero saves city"
Doc 2: "hero fights villain"
Doc 3: "romantic love story"
Doc 4: "hero adventure quest"
Doc 5: "love romance comedy"
```

N = 5 documents.

```
Word        df(t)    IDF = log(5 / (1 + df))
──────────  ─────    ────────────────────────
"hero"        3      log(5 / 4) = log(1.25)  = 0.223
"love"        2      log(5 / 3) = log(1.67)  = 0.511
"action"      1      log(5 / 2) = log(2.50)  = 0.916
"romantic"    1      log(5 / 2) = log(2.50)  = 0.916
"city"        1      log(5 / 2) = log(2.50)  = 0.916
```

Observe:

- "hero" appears in 3 out of 5 documents --> low IDF (0.223). It is common.
- "action" appears in only 1 document --> high IDF (0.916). It is distinctive.

If a word appeared in all 5 documents: `log(5/6) = log(0.833) = -0.182` (negative!).
This is why scikit-learn adds `1 +` smoothing -- to keep values positive.

> **scikit-learn's actual formula** (with smoothing):
> `IDF(t) = log((1 + N) / (1 + df(t))) + 1`
> The `+ 1` at the end ensures that words appearing in every document still get a
> non-zero weight rather than being completely eliminated.

---

### 3.3 Combined TF-IDF

The final TF-IDF weight for a word in a document is simply:

```
TF-IDF(t, d) = TF(t, d) x IDF(t)
```

**Full worked numeric example:**

Corpus (3 documents):

```
Doc 1: "space adventure space exploration"    (4 words)
Doc 2: "space comedy funny comedy"            (4 words)
Doc 3: "romantic comedy love"                 (3 words)
```

N = 3 documents.

**Step 1: Compute TF (raw counts for simplicity)**

```
              space  adventure  exploration  comedy  funny  romantic  love
Doc 1:   [     2,      1,          1,          0,     0,      0,       0  ]
Doc 2:   [     1,      0,          0,          2,     1,      0,       0  ]
Doc 3:   [     0,      0,          0,          1,     0,      1,       1  ]
```

**Step 2: Compute IDF**

```
Word           df(t)    IDF = log(3 / (1+df))
───────────    ─────    ──────────────────────
space            2      log(3/3) = log(1.0)   = 0.000
adventure        1      log(3/2) = log(1.5)   = 0.405
exploration      1      log(3/2) = log(1.5)   = 0.405
comedy           2      log(3/3) = log(1.0)   = 0.000
funny            1      log(3/2) = log(1.5)   = 0.405
romantic         1      log(3/2) = log(1.5)   = 0.405
love             1      log(3/2) = log(1.5)   = 0.405
```

"space" and "comedy" both appear in 2 of 3 documents --> IDF = 0.0 (they are common).
All other words appear in only 1 document --> IDF = 0.405 (they are distinctive).

**Step 3: Multiply TF x IDF**

```
                space    adventure  exploration  comedy   funny   romantic  love
Doc 1:   [  2x0.000,   1x0.405,    1x0.405,   0x0.000, 0x0.405, 0x0.405, 0x0.405 ]
       = [    0.000,     0.405,      0.405,     0.000,   0.000,   0.000,   0.000   ]

Doc 2:   [  1x0.000,   0x0.405,    0x0.405,   2x0.000, 1x0.405, 0x0.405, 0x0.405 ]
       = [    0.000,     0.000,      0.000,     0.000,   0.405,   0.000,   0.000   ]

Doc 3:   [  0x0.000,   0x0.405,    0x0.405,   1x0.000, 0x0.405, 1x0.405, 1x0.405 ]
       = [    0.000,     0.000,      0.000,     0.000,   0.000,   0.405,   0.405   ]
```

Look what happened:

- "space" appeared twice in Doc 1, but its TF-IDF is **0.000** because it is too
  common (appears in 2/3 documents). TF-IDF killed it.
- "adventure" only appeared once in Doc 1, but its TF-IDF is **0.405** because it
  is rare. TF-IDF boosted it.
- The result: each document is now represented by the words that make it _unique_,
  not the words that are common across documents.

---

### 3.4 Sublinear TF (`sublinear_tf=True`)

In our project's `predict_model.py`, the TfidfVectorizer for `overview` and `keywords`
uses `sublinear_tf=True`:

```python
# From predict_model.py line 24
'overview': ('overview_clean', TfidfVectorizer, {
    'sublinear_tf': True,
    'max_features': 15000,
    'stop_words': 'english'
}),
```

**What does `sublinear_tf=True` do?**

Instead of using the raw term frequency, it applies a logarithmic transformation:

```
Standard TF:     TF(t, d) = count of t in d

Sublinear TF:    TF(t, d) = 1 + log(count of t in d)     if count > 0
                 TF(t, d) = 0                              if count = 0
```

**Why? The principle of diminishing returns.**

Consider a movie overview where "action" appears 10 times versus one where it
appears once. Is the first movie really _10 times more_ about action? Probably not.
Seeing "action" once tells you the movie is about action. Seeing it 10 times does
not make it 10x more relevant -- it only adds a little more confidence.

The logarithm compresses large counts:

```
Raw count:     1     2      3      5      10      20      100
Sublinear:    1.0   1.69   2.10   2.61   3.30    4.00    5.61
```

```
                                                       ......... 100 --> 5.61
          ......... 10 --> 3.30                  .....
     ... 3 --> 2.10                          ...
   .. 2 --> 1.69                          ..
  . 1 --> 1.0                          ...         <-- sublinear TF
                                    ...
 ─┬────┬────┬────┬──              .
  1    2    3    5            ...
                           ..
                        ...                        <-- raw count TF (straight line)
                     ...
```

The raw count is a straight line: 10x the count = 10x the TF.
The sublinear version curves and flattens: 10x the count = only ~3.3x the TF.

**Numeric comparison:**

```
Word "action" appears 10 times in a document.

  Standard TF-IDF:   10  x  IDF("action")  =  10  x  0.916  =  9.16
  Sublinear TF-IDF:  (1 + log(10)) x IDF   =  3.30 x  0.916  =  3.02
```

Sublinear TF prevents a document from being dominated by a single repeated word.

---

### 3.5 Stop Words Removal

In our project, the overview vectorizer uses `stop_words='english'`:

```python
'overview': ('overview_clean', TfidfVectorizer, {
    'sublinear_tf': True,
    'max_features': 15000,
    'stop_words': 'english'        # <--- this
}),
```

**What are stop words?**

Stop words are extremely common words that carry little to no meaning on their own.
English examples:

```
the, a, an, is, are, was, were, be, been, being,
have, has, had, do, does, did, will, would, shall,
should, may, might, must, can, could, of, at, by,
for, with, about, against, between, through, during,
before, after, above, below, to, from, up, down, in,
out, on, off, over, under, again, further, then, once,
he, she, it, they, them, their, this, that, these, those,
i, me, my, we, our, you, your, and, but, or, not, no, ...
```

**Why remove them?**

Even though TF-IDF already penalizes common words (via low IDF), stop words are so
overwhelmingly common that they still consume vocabulary space and add noise. Removing
them outright is cleaner and faster.

Consider a movie overview:

> "The young hero must find the courage to save the world from the darkness."

Without stop word removal, the vocabulary includes: the (x4), to, must, from.
These words waste 7 vector dimensions that will carry near-zero TF-IDF values anyway.
By removing them upfront, we get a more compact, meaningful vocabulary:

> "young hero find courage save world darkness"

**Why only for overviews?** The genres, cast, and director fields are not natural
English text -- they are things like "sciencefiction" or "christophernolan." There
are no stop words to remove.

---

## 4. CountVectorizer vs TfidfVectorizer

scikit-learn provides two main text vectorizers. Our project uses **both**, for
different fields.

### CountVectorizer

CountVectorizer implements basic Bag of Words. It produces **raw word counts**.

```
Input:  "action comedy action"
Output: [2, 1]   (action appears 2x, comedy 1x)
```

**Used in our project for:**

```python
# From predict_model.py lines 25-31
'genres':     ('genres_str',     CountVectorizer, {}),
'cast':       ('cast_str',       CountVectorizer, {}),
'director':   ('director_str',   CountVectorizer, {}),
'decade':     ('decade',         CountVectorizer, {}),
'language':   ('language',       CountVectorizer, {}),
'collection': ('collection',     CountVectorizer, {}),
```

**Why CountVectorizer for these fields?**

These fields are short, categorical, and structured:

```
genres_str:    "action adventure sciencefiction"     (3 tokens max ~5)
cast_str:      "robertdowneyjr gwynethpaltrow"       (cleaned names, ~5 tokens)
director_str:  "jonfavreau"                          (single token)
decade:        "2000s"                               (single token)
```

There is no concept of "a common word drowning out a rare word" here. Each genre
either applies to this movie or it does not. Each director is a single token. Raw
counts (which are basically 0 or 1 for these short fields) are perfectly adequate.
TF-IDF's sophisticated weighting would add complexity without benefit.

### TfidfVectorizer

TfidfVectorizer performs the full TF-IDF calculation: term frequency, inverse document
frequency, and L2 normalization.

**Used in our project for:**

```python
# From predict_model.py lines 24, 26
'overview':  ('overview_clean', TfidfVectorizer, {
    'sublinear_tf': True, 'max_features': 15000, 'stop_words': 'english'
}),
'keywords':  ('keywords_str',  TfidfVectorizer, {
    'sublinear_tf': True, 'max_features': 5000
}),
```

**Why TfidfVectorizer for these fields?**

Overviews are natural language text, often 50-200 words long. They contain common
words ("the," "and"), filler words, and genuinely informative words mixed together.
TF-IDF is essential to separate signal from noise.

Keywords are shorter but can still have common-vs-rare distinctions. A keyword like
"murder" appears in hundreds of movies, while "timeloop" appears in only a few.
TF-IDF correctly downweights the common keywords.

### Summary Table

```
┌─────────────┬──────────────────────┬────────────────────────────────────────┐
│ Field       │ Vectorizer           │ Why                                    │
├─────────────┼──────────────────────┼────────────────────────────────────────┤
│ overview    │ TfidfVectorizer      │ Long text, needs word importance       │
│ keywords    │ TfidfVectorizer      │ Variable importance across keywords    │
│ genres      │ CountVectorizer      │ Short categorical, 0/1 is sufficient   │
│ cast        │ CountVectorizer      │ Cleaned names, each is a single token  │
│ director    │ CountVectorizer      │ Single token per movie                 │
│ decade      │ CountVectorizer      │ Single token ("2000s")                 │
│ language    │ CountVectorizer      │ Single token ("en")                    │
│ collection  │ CountVectorizer      │ Single token or empty                  │
└─────────────┴──────────────────────┴────────────────────────────────────────┘
```

---

## 5. The Multi-Vectorizer Architecture

Most simple recommendation tutorials use a single TF-IDF vectorizer on a concatenated
"soup" of all metadata. Our project uses a more sophisticated approach: **separate
vectorizers per field, with configurable weights, combined into one feature matrix.**

This is defined in `src/models/predict_model.py`.

### 5.1 FIELD_CONFIG: The Blueprint

```python
FIELD_CONFIG: dict[str, tuple[str, type, dict]] = {
    'overview':   ('overview_clean', TfidfVectorizer,  {'sublinear_tf': True, 'max_features': 15000, 'stop_words': 'english'}),
    'genres':     ('genres_str',     CountVectorizer,   {}),
    'keywords':   ('keywords_str',   TfidfVectorizer,  {'sublinear_tf': True, 'max_features': 5000}),
    'cast':       ('cast_str',       CountVectorizer,   {}),
    'director':   ('director_str',   CountVectorizer,   {}),
    'decade':     ('decade',         CountVectorizer,   {}),
    'language':   ('language',       CountVectorizer,   {}),
    'collection': ('collection',     CountVectorizer,   {}),
}
```

Each entry is:

```
field_name: (dataframe_column, vectorizer_class, vectorizer_settings)
```

For example, `'overview'` says:

- Read the `overview_clean` column from the DataFrame
- Use `TfidfVectorizer` to vectorize it
- Pass `sublinear_tf=True`, `max_features=15000`, `stop_words='english'` to the vectorizer

`max_features=15000` means: keep only the 15,000 most frequent terms in the vocabulary.
This limits the vector size and discards extremely rare words.

### 5.2 Per-Field Vectorization

The `fit()` method loops over every field and creates a separate matrix:

```python
def fit(self) -> 'ContentBasedRecommender':
    matrices = []

    for field, (col, vec_cls, vec_kwargs) in FIELD_CONFIG.items():
        weight = self.weights.get(field, 0.0)
        if weight == 0.0:
            continue

        texts = self.smd[col].fillna('').astype(str)
        vectorizer = vec_cls(**vec_kwargs)
        matrix = vectorizer.fit_transform(texts)

        if weight != 1.0:
            matrix = matrix * weight

        self.vectorizers[field] = vectorizer
        matrices.append(matrix)

    self.feature_matrix = hstack(matrices, format='csr')
    self.cosine_sim = cosine_similarity(self.feature_matrix)
```

Each field produces its own matrix. Let us visualize what happens with a tiny
example (4 movies):

```
OVERVIEW vectorizer produces:
┌──────────────────────────────────────────────────┐
│  4 movies x 15000 features (max_features=15000)  │
│  (In practice maybe ~8000 non-empty features)    │
└──────────────────────────────────────────────────┘

GENRES vectorizer produces:
┌────────────────────┐
│  4 movies x ~20    │
│  (there are ~20    │
│   unique genres)   │
└────────────────────┘

KEYWORDS vectorizer produces:
┌──────────────────────────────────┐
│  4 movies x 5000 features       │
└──────────────────────────────────┘

CAST vectorizer produces:
┌──────────────────────────────────┐
│  4 movies x ~3000 features      │
│  (unique actor tokens)          │
└──────────────────────────────────┘

DIRECTOR vectorizer produces:
┌──────────────────────────────────┐
│  4 movies x ~1500 features      │
│  (unique director tokens)       │
└──────────────────────────────────┘

DECADE vectorizer produces:
┌─────────────────┐
│  4 movies x ~13 │
│  (1920s..2020s) │
└─────────────────┘

LANGUAGE vectorizer produces:
┌─────────────────┐
│  4 movies x ~30 │
│  (unique langs) │
└─────────────────┘

COLLECTION vectorizer produces:
┌──────────────────────────────────┐
│  4 movies x ~500 features       │
│  (unique collection names)      │
└──────────────────────────────────┘
```

Each matrix has the **same number of rows** (one per movie) but a **different number
of columns** (the vocabulary size of that field).

### 5.3 Field Weighting

Before combining, each matrix is multiplied by its field weight:

```python
DEFAULT_WEIGHTS: dict[str, float] = {
    'overview': 1.0,
    'genres': 1.5,
    'keywords': 1.2,
    'cast': 1.0,
    'director': 2.0,     # <-- highest weight
    'decade': 0.3,       # <-- lowest weight
    'language': 0.5,
    'collection': 1.5,
}
```

```python
if weight != 1.0:
    matrix = matrix * weight   # scalar multiplication on the entire matrix
```

This means every value in the director matrix gets multiplied by 2.0, making director
matches count twice as much as overview matches. Every value in the decade matrix gets
multiplied by 0.3, making decade matches count far less.

**Why these weights?**

- `director = 2.0`: Two movies by the same director are strongly likely to be
  similar in style (e.g., all Christopher Nolan films share a certain quality).
  This is the strongest signal.
- `genres = 1.5`: Genre match is very important for recommendations.
- `collection = 1.5`: Movies in the same collection (e.g., "Harry Potter Collection")
  are obviously similar.
- `keywords = 1.2`: Keywords like "dystopia" or "timetravel" are meaningful but
  noisier than genres.
- `overview = 1.0`: The overview is informative but wordy. Baseline weight.
- `cast = 1.0`: Shared actors are a moderate signal.
- `language = 0.5`: Same language is a weak signal (most movies are English anyway).
- `decade = 0.3`: Same decade is the weakest signal. A 2010s action movie and a
  2010s romance have little in common just because of the decade.

### 5.4 Horizontal Stacking with `hstack`

After weighting, all matrices are combined side-by-side using `hstack`:

```python
self.feature_matrix = hstack(matrices, format='csr')
```

`hstack` = **horizontal stack**. It concatenates matrices column-wise.

Visually, for one movie:

```
overview features    genres   keywords features   cast features    dir   decade lang  coll
[0.2, 0.0, ..., 0.1, 1.5, 0, 1.5, 0, 0.3, 0.0, ..., 0.4, 0, ..., 2.0, 0.3,  0.5,  0  ]
|<--- 15000 --->|  |<-20->|  |<-- 5000 -->|  |<- ~3000 ->| |~1500| |13| |30| |~500|
|__________________|________|______________|______________|_______|____|____|_____|
                                    |
                          One combined feature vector
                        ~25,000+ dimensions per movie
```

The final feature matrix has shape:

```
(number_of_movies, sum_of_all_vocabulary_sizes)
```

For our project with ~9000 movies:

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│                  ~25,000+ columns                        │
│              (one per unique token                       │
│               across all fields)                         │
│                                                          │
│  ~9000   ┌───────────────────────────────────────────┐   │
│  rows    │                                           │   │
│  (one    │     Combined feature matrix               │   │
│  per     │                                           │   │
│  movie)  │     Each cell: weighted TF-IDF or         │   │
│          │     weighted count value                   │   │
│          │                                           │   │
│          │     Most cells are 0.0 (sparse!)          │   │
│          │                                           │   │
│          └───────────────────────────────────────────┘   │
│                                                          │
│  Shape: approximately (9000, 25000)                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 5.5 What Each Dimension Represents

Every column in the final matrix corresponds to a specific token from a specific field:

```
Column 0:        overview word "abandoned"
Column 1:        overview word "ability"
...
Column 14999:    overview word "zombie"
Column 15000:    genre token "action"
Column 15001:    genre token "adventure"
...
Column 15019:    genre token "western"
Column 15020:    keyword token "alien"
...
Column 20019:    keyword token "zombie"
Column 20020:    cast token "aaaborham"
...
Column 23019:    cast token "zoesaldana"
Column 23020:    director token "aaronsorkin"
...
Column 24519:    director token "zacksnyder"
Column 24520:    decade token "1950s"
...
Column 24532:    decade token "2020s"
Column 24533:    language token "en"
...
Column 24562:    language token "zh"
Column 24563:    collection token "harrypottercollection"
...
Column 25062:    collection token "xmencollection"
```

(Numbers are approximate -- the actual vocabulary sizes depend on the data.)

When two movies share the same director, they both have a non-zero value in the same
"director token" column. Because that column is weighted by 2.0, this shared director
contributes significantly to their overall similarity score.

### 5.6 Why This Multi-Vectorizer Approach?

Compared to the simpler "concatenate everything into one soup and use a single
TfidfVectorizer," this architecture has several advantages:

1. **Different vectorizers per field type.** Genres do not need IDF weighting, but
   overviews do. We can use CountVectorizer where it is appropriate and
   TfidfVectorizer where it is needed.

2. **Different vocabulary settings.** We can limit the overview vocabulary to 15,000
   terms while giving keywords a separate 5,000-term vocabulary. In a single
   vectorizer, rare keywords might get crowded out by common overview words.

3. **Adjustable field weights.** We can tune how much director, genre, or overview
   matter independently. In a soup approach, you would repeat words (e.g., paste the
   director 3 times) -- which is a crude hack.

4. **Clean separation of concerns.** Each field is vectorized independently. Changing
   the director weight does not require re-vectorizing the overview.

---

## 6. Sparse Matrices

### The Problem: Most Values Are Zero

Consider our feature matrix of shape (9000, 25000). That is 225 million cells.
How many of them are non-zero?

A single movie's overview might use 50 unique words out of a 15,000-word vocabulary.
Its genres might activate 3 out of 20 genre slots. Its cast activates 5 out of 3,000
actor slots. Its director activates 1 out of 1,500 director slots.

Total non-zero values per movie: roughly 50 + 3 + 10 + 5 + 1 + 1 + 1 + 1 = ~72

Out of ~25,000 columns, only ~72 are non-zero. That is **0.3%** non-zero.
The matrix is **99.7% zeros.**

### Why Not Use a Normal Matrix?

A normal (dense) NumPy matrix stores every single value:

```
Dense matrix (9000 x 25000):
  Memory = 9000 x 25000 x 8 bytes (float64) = 1.8 GB

  Stores: [0.0, 0.0, 0.0, 0.41, 0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.58, 0.0, ...]
                                                        ^^^^^^^^^^^^^^^^^^^
                                                        all these zeros waste memory
```

1.8 GB for a matrix that is 99.7% zeros. That is enormously wasteful.

### Sparse Matrices: Store Only Non-Zero Values

A sparse matrix stores only the values that are not zero, along with their positions.

scipy provides several sparse matrix formats. Our project uses **CSR (Compressed
Sparse Row)**, specified in the `hstack` call:

```python
self.feature_matrix = hstack(matrices, format='csr')
```

### How CSR Works

CSR stores three arrays:

- `data`: the actual non-zero values
- `indices`: the column index for each non-zero value
- `indptr`: where each row's data starts and ends

**Example:** A 3x5 matrix with only 4 non-zero values:

```
Dense representation:
     col0  col1  col2  col3  col4
row0 [ 0.0,  0.4,  0.0,  0.0,  0.0 ]
row1 [ 0.0,  0.0,  0.3,  0.0,  0.7 ]
row2 [ 0.5,  0.0,  0.0,  0.0,  0.0 ]

CSR representation:
  data    = [0.4, 0.3, 0.7, 0.5]     -- the non-zero values
  indices = [ 1,   2,   4,   0 ]     -- which column each value is in
  indptr  = [ 0,   1,   3,   4 ]     -- row boundaries in data array
                |    |    |    |
                |    |    |    └─ row2 data ends at position 4
                |    |    └─ row1 data ends at position 3 (row2 starts here)
                |    └─ row0 data ends at position 1 (row1 starts here)
                └─ row0 data starts at position 0
```

Reading row 1: `indptr[1]=1` to `indptr[2]=3`, so data[1:3] = [0.3, 0.7] at
columns indices[1:3] = [2, 4].

### Memory Savings

```
Dense (9000 x 25000):
  9000 x 25000 x 8 bytes = ~1.8 GB

Sparse CSR (same matrix, ~0.3% non-zero):
  Non-zero values: ~9000 x 72 = ~648,000 entries
  data array:     648,000 x 8 bytes  =  ~5.2 MB
  indices array:  648,000 x 4 bytes  =  ~2.6 MB
  indptr array:   9,001 x 4 bytes    =  ~0.04 MB
  Total:                              =  ~7.8 MB
```

```
Dense:   ████████████████████████████████████████  1,800 MB
Sparse:  █                                            8 MB
```

That is a **230x** reduction in memory. This is why the project uses sparse matrices
throughout the vectorization pipeline -- both `TfidfVectorizer` and `CountVectorizer`
return sparse matrices by default, `hstack` keeps them sparse, and `cosine_similarity`
can accept sparse input.

### In the Code

The entire flow stays sparse until cosine similarity:

```python
# Each vectorizer returns a sparse matrix
matrix = vectorizer.fit_transform(texts)      # sparse (9000 x vocab_size)

# Scalar multiplication preserves sparsity
matrix = matrix * weight                      # still sparse

# hstack combines sparse matrices, result is sparse
self.feature_matrix = hstack(matrices, format='csr')   # sparse (9000 x ~25000)

# cosine_similarity accepts sparse input but returns dense (it has to --
# the similarity matrix between 9000 movies is 9000x9000 = 81 million values,
# and most of those are NOT zero because most movies have at least tiny similarity)
self.cosine_sim = cosine_similarity(self.feature_matrix)  # dense (9000 x 9000)
```

---

## Summary: The Full Picture

Here is the complete text-to-numbers pipeline as it runs in our project:

```
Raw movie data
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  build_features() -- src/features/build_features.py     │
│                                                         │
│  Clean text fields:                                     │
│    overview  → lowercase                                │
│    genres    → "action adventure sciencefiction"         │
│    keywords  → "dystopia timetravel"                    │
│    cast      → "keanureeves laurencefishburne"          │
│    director  → "thewachowskis"                          │
│    decade    → "1990s"                                  │
│    language  → "en"                                     │
│    collection→ "thematrixcollection"                    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  fit() -- src/models/predict_model.py                   │
│                                                         │
│  For each field:                                        │
│    1. Choose vectorizer (TF-IDF or Count)               │
│    2. Fit vocabulary on all movies                      │
│    3. Transform text → sparse matrix                    │
│    4. Multiply by field weight                          │
│                                                         │
│  overview   → TF-IDF  → (9000 x 15000) x 1.0           │
│  genres     → Count   → (9000 x 20)    x 1.5           │
│  keywords   → TF-IDF  → (9000 x 5000)  x 1.2           │
│  cast       → Count   → (9000 x 3000)  x 1.0           │
│  director   → Count   → (9000 x 1500)  x 2.0           │
│  decade     → Count   → (9000 x 13)    x 0.3           │
│  language   → Count   → (9000 x 30)    x 0.5           │
│  collection → Count   → (9000 x 500)   x 1.5           │
│                                                         │
│  hstack all → (9000 x ~25000) sparse feature matrix     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  cosine_similarity(feature_matrix)                      │
│                                                         │
│  → (9000 x 9000) similarity matrix                     │
│                                                         │
│  Entry [i][j] = how similar movie i is to movie j       │
│  Values range from 0.0 (nothing in common)              │
│                   to 1.0 (identical feature vectors)    │
└─────────────────────────────────────────────────────────┘
```

This is how text becomes numbers in our movie recommendation system. Every movie ends
up as a long vector of ~25,000 numbers, where each number captures how important a
specific word or token is for that movie, weighted by how distinctive that word is
across all movies and how important that field is for recommendations.
