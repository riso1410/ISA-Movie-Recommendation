# Data Preparation for a Content-Based Movie Recommendation System

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [What Raw Data We Start With](#2-what-raw-data-we-start-with)
3. [Data Loading and Cleaning (make_dataset.py)](#3-data-loading-and-cleaning)
4. [Feature Engineering (build_features.py)](#4-feature-engineering)
5. [Poster Fetching (fetch_posters.py)](#5-poster-fetching)
6. [Summary: The Complete Pipeline](#6-summary-the-complete-pipeline)

---

## 1. The Big Picture

### What is a content-based recommendation system?

Imagine you walk into a video store and tell the clerk: "I loved _The Dark Knight_ -- what else
should I watch?" A good clerk would think about what makes that movie special: it is a
**crime/action/drama**, it has **Christopher Nolan** as director, it stars **Christian Bale** and
**Heath Ledger**, and its plot is about a **vigilante** fighting **organized crime** in a city.
The clerk would then suggest other movies that share those characteristics.

A **content-based recommendation system** does exactly this, but with math. It looks at the
_content_ (attributes) of movies you like and finds other movies with similar content.

The core idea:

```
Movies you liked ──> Extract what makes them special ──> Find similar movies ──> Recommend
```

But before we can do any of that, we need to prepare the data. Raw data from the internet is
messy, spread across multiple files, and stored in formats that machines cannot directly compare.
This document explains every step of turning raw, messy CSV files into clean, structured data
that a recommendation algorithm can work with.

### The Full Pipeline at a Glance

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RAW DATA (5 CSV files)                        │
│  movies_metadata.csv  credits.csv  keywords.csv  links_small.csv     │
│  ratings_small.csv                                                   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   make_dataset.py       │
                    │   (Load, Clean, Merge)  │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   build_features.py     │
                    │   (Feature Engineering) │
                    └────────────┬───────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │  movies_processed.csv                  │
              │  (Clean, merged, feature-engineered)   │
              └──────────────────────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │  Recommendation Algorithm             │
              │  (TF-IDF + Cosine Similarity)         │
              └──────────────────────────────────────┘
```

---

## 2. What Raw Data We Start With

The raw data comes from the **"The Movies Dataset"** on Kaggle. It is based on data from
**TMDB** (The Movie Database) and **MovieLens**, and is split across five CSV files. Think of
each file as a separate spreadsheet, each holding a different aspect of movie information.

### 2.1 movies_metadata.csv -- The Core Movie Information

This is the main file. Each row is one movie.

| Column                  | What It Contains                            | Example                                              |
| ----------------------- | ------------------------------------------- | ---------------------------------------------------- |
| `id`                    | TMDB movie ID (unique identifier)           | `862`                                                |
| `title`                 | Movie title                                 | `Toy Story`                                          |
| `overview`              | Plot summary paragraph                      | `Led by Woody, Andy's toys live happily...`          |
| `genres`                | List of genres (stored as JSON string)      | `[{"id": 16, "name": "Animation"}, ...]`             |
| `original_language`     | Language code                               | `en`                                                 |
| `release_date`          | Release date                                | `1995-10-30`                                         |
| `vote_average`          | Average user rating (0-10)                  | `7.7`                                                |
| `vote_count`            | Number of user votes                        | `5415`                                               |
| `popularity`            | TMDB popularity score                       | `21.946943`                                          |
| `belongs_to_collection` | Movie franchise/series (JSON string or NaN) | `{"id": 10194, "name": "Toy Story Collection", ...}` |
| ...                     | Other columns (budget, revenue, etc.)       | (not used in our system)                             |

**Why we need it:** This file provides the foundational information about every movie -- title,
plot summary, genres, ratings, and popularity. The overview text and genre tags are critical
ingredients for comparing movies to each other.

**Key detail:** The `genres` and `belongs_to_collection` columns are stored as _JSON strings_
inside CSV cells. This means what looks like a list of dictionaries is actually just plain text
that needs to be parsed. More on this later.

### 2.2 credits.csv -- Who Made Each Movie

Each row corresponds to one movie and contains the full cast and crew information.

| Column | What It Contains                             | Example                                                    |
| ------ | -------------------------------------------- | ---------------------------------------------------------- |
| `id`   | TMDB movie ID (links to movies_metadata)     | `862`                                                      |
| `cast` | Full cast list (JSON string of dictionaries) | `[{"name": "Tom Hanks", "character": "Woody", ...}, ...]`  |
| `crew` | Full crew list (JSON string of dictionaries) | `[{"name": "John Lasseter", "job": "Director", ...}, ...]` |

**Why we need it:** The director and lead actors are powerful signals for recommendation. If
you liked a Christopher Nolan film, you might like other Christopher Nolan films. If you
enjoyed Tom Hanks in _Forrest Gump_, you might enjoy Tom Hanks in _Cast Away_.

**Key detail:** The `cast` column contains _everyone_ who appeared in the movie -- sometimes
dozens of people. We only extract the **top 5** cast members (the most prominent actors),
because minor roles add noise without adding useful signal.

**Key detail:** The `crew` column contains everyone who worked on the movie (director,
producers, cinematographers, editors, etc.). We only extract the person whose `job` field
equals `"Director"`.

### 2.3 keywords.csv -- Thematic Tags

Each row corresponds to one movie with a list of descriptive tags.

| Column     | What It Contains               | Example                                                               |
| ---------- | ------------------------------ | --------------------------------------------------------------------- |
| `id`       | TMDB movie ID                  | `862`                                                                 |
| `keywords` | Descriptive tags (JSON string) | `[{"id": 931, "name": "jealousy"}, {"id": 4290, "name": "toy"}, ...]` |

**Why we need it:** Keywords capture _thematic_ content that the genre alone misses. Two movies
might both be "Action" films, but one is about "space travel" and "aliens" while the other is
about "martial arts" and "revenge". Keywords provide this fine-grained thematic information.

### 2.4 links_small.csv -- The MovieLens Subset Filter

| Column    | What It Contains                         | Example  |
| --------- | ---------------------------------------- | -------- |
| `movieId` | MovieLens movie ID                       | `1`      |
| `imdbId`  | IMDB movie ID                            | `114709` |
| `tmdbId`  | TMDB movie ID (links to movies_metadata) | `862`    |

**Why we need it:** The full movies_metadata.csv contains around **45,000 movies**. That is a
lot of data and most of those movies are obscure titles with very little metadata. This file
acts as a **filter** -- it contains only the ~9,000 movies that are in the MovieLens "small"
dataset. These are popular, well-known movies with rich metadata, which makes for better
recommendations.

Think of it as a curated whitelist: "only use these movies."

### 2.5 ratings_small.csv -- User Rating Data

| Column      | What It Contains               | Example      |
| ----------- | ------------------------------ | ------------ |
| `userId`    | Anonymous user ID              | `1`          |
| `movieId`   | MovieLens movie ID             | `31`         |
| `rating`    | Rating the user gave (0.5-5.0) | `2.5`        |
| `timestamp` | When the rating was made       | `1260759144` |

**Why we need it:** While our system is _content-based_ (not collaborative filtering), the
ratings data is loaded as part of the full pipeline. It can be used for evaluation purposes --
to test whether movies we recommend are ones that users actually rated highly.

### How the Files Connect

```
movies_metadata.csv ◄─── id ───► credits.csv
        │
        │ id
        ▼
   keywords.csv

links_small.csv ─── tmdbId ───► movies_metadata.csv (filter)
        │
        │ movieId
        ▼
ratings_small.csv
```

The `id` column is the glue. Every file uses the TMDB movie ID (`id` or `tmdbId`) to link
information about the same movie across different files.

---

## 3. Data Loading and Cleaning

File: **`src/data/make_dataset.py`**

This script takes the five raw CSV files and produces a single, clean, merged dataset. Let us
walk through it step by step.

### 3.1 Loading the Raw Files

```python
def load_raw_data(raw_dir='data/raw'):
    metadata = pd.read_csv(raw_dir / 'movies_metadata.csv', low_memory=False)
    credits = pd.read_csv(raw_dir / 'credits.csv')
    keywords = pd.read_csv(raw_dir / 'keywords.csv')
    links_small = pd.read_csv(raw_dir / 'links_small.csv')
    ratings = pd.read_csv(raw_dir / 'ratings_small.csv')
    return metadata, credits, keywords, links_small, ratings
```

Each `pd.read_csv()` call reads one CSV file into a pandas DataFrame (think of it as a
spreadsheet in memory). The `low_memory=False` flag on movies_metadata tells pandas to read
the entire file before guessing column types, which avoids mixed-type warnings for this
particular file.

### 3.2 Cleaning Metadata IDs

```python
metadata = metadata[metadata['id'].apply(lambda x: str(x).isdigit())].copy()
metadata['id'] = metadata['id'].astype(int)
```

**The problem:** The `id` column in movies_metadata.csv _should_ contain numeric IDs like
`862`, `8844`, `15602`. But the raw data has a few corrupted rows where the `id` field
contains non-numeric garbage values (dates, text strings, etc. that leaked into the wrong
column due to CSV formatting issues in the original dataset).

**The fix:** We check every ID -- is it made entirely of digits? If not, throw that row out.
Then convert all remaining IDs to integers so they can be compared with IDs from other files.

**Concrete example:**

```
Before filtering:
  id = "862"      ──> keeps (numeric)
  id = "8844"     ──> keeps (numeric)
  id = "1997-08-20" ──> REMOVED (not numeric -- corrupted row)
  id = "False"    ──> REMOVED (not numeric -- corrupted row)

After filtering:
  id = 862   (integer)
  id = 8844  (integer)
```

### 3.3 Filtering to the links_small Subset

```python
links_small = links_small[links_small['tmdbId'].notna()].copy()
links_small['tmdbId'] = links_small['tmdbId'].astype(int)
smd = metadata[metadata['id'].isin(links_small['tmdbId'])].copy()
```

**What this does:** We only keep movies whose TMDB ID appears in the `links_small.csv` file.

**Why:** As mentioned earlier, the full dataset has ~45,000 movies. Many are obscure foreign
films, short films, or entries with almost no metadata. The links_small subset narrows this
down to ~9,000 well-known movies with rich metadata. Working with a smaller, higher-quality
dataset produces better recommendations and is faster to process.

```
Full metadata:  ~45,000 movies
                    │
                    │  Filter by links_small.tmdbId
                    ▼
Filtered (smd):  ~9,000 movies  (small metadata dataset)
```

The variable name `smd` stands for "small metadata" -- a common convention in tutorials based
on this dataset.

### 3.4 Parsing JSON Columns with literal_eval

This is one of the trickiest parts of working with this dataset.

**The problem:** CSV files store everything as plain text. When the original data was exported,
complex structures like lists of dictionaries were converted to their _string representation_.
So a cell that should contain a Python list actually contains the text characters that spell
out a list.

Here is what a `genres` cell actually looks like as raw text in the CSV:

```
[{"id": 16, "name": "Animation"}, {"id": 35, "name": "Comedy"}, {"id": 10751, "name": "Family"}]
```

This is NOT a Python list. It is a _string_ that looks like a Python list. If you try to
iterate over it or access elements, Python will treat it as a sequence of characters, not as
a data structure.

**The fix:** `literal_eval` from Python's `ast` module safely parses a string that looks like
a Python literal and converts it into the actual Python object:

```python
from ast import literal_eval

raw_text = '[{"id": 16, "name": "Animation"}, {"id": 35, "name": "Comedy"}]'
parsed = literal_eval(raw_text)
# parsed is now an actual Python list of dictionaries:
# [{"id": 16, "name": "Animation"}, {"id": 35, "name": "Comedy"}]

# Now we can extract just the genre names:
genre_names = [d['name'] for d in parsed]
# genre_names = ["Animation", "Comedy"]
```

This pattern (literal_eval + extract names) is used for four columns:

| Column     | Raw CSV text                                      | After literal_eval + extraction     |
| ---------- | ------------------------------------------------- | ----------------------------------- |
| `genres`   | `[{"id":16, "name":"Animation"}, ...]`            | `["Animation", "Comedy", "Family"]` |
| `keywords` | `[{"id":931, "name":"jealousy"}, ...]`            | `["jealousy", "toy", "boy"]`        |
| `cast`     | `[{"name":"Tom Hanks", ...}, ...]`                | `["Tom Hanks", "Tim Allen", ...]`   |
| `crew`     | `[{"name":"John Lasseter","job":"Director"},...]` | `"John Lasseter"` (director only)   |

### 3.5 Parsing Credits: Extracting Cast and Director

```python
def parse_credits(credits_df):
    cred['cast'] = cred['cast'].apply(
        lambda x: [d['name'] for d in literal_eval(x)][:5]
    )
    cred['director'] = cred['crew'].apply(
        lambda x: next(
            (d['name'] for d in literal_eval(x) if d['job'] == 'Director'), np.nan
        )
    )
    return cred[['id', 'cast', 'director']]
```

**Cast extraction -- step by step:**

1. Take the raw `cast` string: `'[{"name": "Tom Hanks", "character": "Woody", "order": 0}, {"name": "Tim Allen", "character": "Buzz Lightyear", "order": 1}, ... 20 more entries ...]'`
2. `literal_eval(x)` turns it into an actual Python list of dictionaries
3. `[d['name'] for d in ...]` extracts just the `name` field from each dictionary
4. `[:5]` takes only the **first 5 names** (the most prominent actors)

**Why only the top 5?** The cast list in the raw data includes every actor with a credited
role -- sometimes 30+ people. Minor actors who appeared in a single scene add noise. The first
5 entries are the lead and main supporting actors, which are the strongest signal for
recommendation. If you liked a movie "because of Tom Hanks," it is because he was a lead, not
an extra.

**Director extraction -- step by step:**

1. Take the raw `crew` string (same format as cast, but with `job` fields)
2. `literal_eval(x)` parses the string into a list of dictionaries
3. `next(d['name'] for d in ... if d['job'] == 'Director')` scans through the list and finds the first person whose `job` is `"Director"`
4. If no director is found, it defaults to `np.nan` (missing value)

**Concrete example for _Toy Story_ (id=862):**

```
Raw crew string (abbreviated):
[
  {"name": "John Lasseter", "job": "Director"},
  {"name": "Joss Whedon",   "job": "Screenplay"},
  {"name": "Joel Cohen",    "job": "Screenplay"},
  ...30 more crew members...
]

After parsing:
  cast = ["Tom Hanks", "Tim Allen", "Don Rickles", "Jim Varney", "Wallace Shawn"]
  director = "John Lasseter"
```

### 3.6 Parsing Keywords

```python
def parse_keywords(keywords_df):
    kw['keywords'] = kw['keywords'].apply(
        lambda x: [d['name'] for d in literal_eval(x)]
    )
    return kw[['id', 'keywords']]
```

This follows the same pattern as genres: parse the JSON string, extract the `name` field from
each dictionary. Unlike cast, we keep _all_ keywords -- they are curated thematic tags so
there is no noisy data to trim.

**Example for _Toy Story_:**

```
Raw: '[{"id": 931, "name": "jealousy"}, {"id": 4290, "name": "toy"}, {"id": 5202, "name": "boy"}, ...]'
After: ["jealousy", "toy", "boy", "friendship", "friends", ...]
```

### 3.7 Deriving Decade, Language, and Collection

These three functions extract additional movie attributes that help the recommendation system:

#### Decade

```python
def parse_decade(date_str: str) -> str:
    year = int(str(date_str)[:4])     # "1995-10-30" -> 1995
    decade = (year // 10) * 10        # 1995 -> 1990
    return f'decade_{decade}s'        # -> "decade_1990s"
```

**Why:** People often have era preferences. If you enjoy 1980s action movies, you might
prefer other 1980s films over a 2020s remake. The decade token lets the system detect this
pattern.

```
"1995-10-30"  ──>  "decade_1990s"
"1972-03-15"  ──>  "decade_1970s"
"2019-10-02"  ──>  "decade_2010s"
```

#### Language

```python
def parse_language(lang: str) -> str:
    return f'lang_{str(lang).strip().lower()}'
```

**Why:** Some viewers have a preference for films in a particular language (English, French,
Korean, etc.). This creates a simple token like `lang_en` or `lang_fr`.

```
"en"  ──>  "lang_en"
"fr"  ──>  "lang_fr"
"ko"  ──>  "lang_ko"
```

#### Collection (Franchise)

```python
def parse_collection(collection_str: str) -> str:
    col = literal_eval(str(collection_str))
    if isinstance(col, dict) and 'name' in col:
        return col['name']
```

**Why:** If you liked _Toy Story_, you probably want to see _Toy Story 2_. The collection
field captures franchise membership.

```
Raw: '{"id": 10194, "name": "Toy Story Collection", "poster_path": "/..."}'
After: "Toy Story Collection"
```

### 3.8 The Merge Process

```python
def merge_datasets(smd, credits_parsed, keywords_parsed):
    smd['genres'] = smd['genres'].fillna('[]').apply(literal_eval).apply(
        lambda x: [d['name'] for d in x] if isinstance(x, list) else []
    )
    smd = smd.merge(keywords_parsed, on='id', how='left')
    smd = smd.merge(credits_parsed,  on='id', how='left')
```

This step combines all the information into a single table.

**What is a merge/join?** Imagine you have two spreadsheets. One has movie titles and genres.
The other has movie IDs and cast lists. A "merge" (or "join") combines them by matching rows
that share the same `id` value, like a VLOOKUP in Excel.

**Why `how='left'`?** A "left join" means: start with every row in the left table (`smd`),
and attach matching data from the right table (`credits_parsed` or `keywords_parsed`). If a
movie exists in `smd` but has no matching entry in the credits file, it stays in the result
with empty/missing values for cast and director -- it does not get thrown out.

```
              LEFT JOIN
  smd (left)          credits_parsed (right)
  ┌──────────┐        ┌───────────────────────────────────────────┐
  │ id: 862  │   ───► │ id: 862, cast: [Tom Hanks,...], dir: ...  │
  │ id: 8844 │   ───► │ id: 8844, cast: [...], dir: ...          │
  │ id: 9999 │   ───► │ (no match -- cast and director are NaN)  │
  └──────────┘        └───────────────────────────────────────────┘

  Result: all 3 movies kept. Movie 9999 has NaN for cast/director.
```

Contrast with an "inner join" which would _discard_ movie 9999 entirely. We use a left join
because we want to keep all movies, even if some metadata is incomplete.

### 3.9 Coercing Numeric Columns

```python
for col in ['vote_average', 'vote_count', 'popularity']:
    smd[col] = pd.to_numeric(smd[col], errors='coerce').fillna(0.0)
```

**Why:** Some values in these columns might be non-numeric (corrupted data, empty strings,
etc.). `pd.to_numeric(..., errors='coerce')` converts valid numbers and turns anything
unparseable into `NaN`. Then `.fillna(0.0)` replaces those NaNs with zero.

This ensures these columns are clean floating-point numbers that can be used in mathematical
calculations (like computing weighted ratings).

### 3.10 Preserving Poster URLs

```python
if out_path.exists():
    existing = pd.read_csv(out_path)
    if 'poster_url' in existing.columns:
        poster_map = existing.drop_duplicates(subset='id').set_index('id')['poster_url']
        smd['poster_url'] = smd['id'].map(poster_map).fillna('')
```

Poster URLs are fetched separately (see Section 5) and take a long time to collect. This
code checks whether a previous version of the processed file already has poster URLs and, if
so, carries them forward into the new output -- so you do not lose hours of poster-fetching
work when you re-run the data pipeline.

### 3.11 The Final Output

The `make_dataset` function saves a single CSV with these columns:

```
id, title, overview, genres, keywords, cast, director,
vote_average, vote_count, popularity, decade, language, collection
```

Each row is one movie, and all the information from all five source files has been merged,
cleaned, and structured.

---

## 4. Feature Engineering

File: **`src/features/build_features.py`**

Feature engineering is the process of transforming raw data into a format that a machine
learning algorithm can work with effectively. The recommendation system uses **TF-IDF**
(explained below) to compare text, so we need to prepare text fields that TF-IDF can process.

### 4.1 Quick Background: What is TF-IDF?

**TF-IDF** (Term Frequency - Inverse Document Frequency) is a way to convert text into
numbers. It answers the question: "How important is this word to this particular document,
compared to all other documents?"

- **TF (Term Frequency):** How often a word appears in this movie's text. If "space" appears
  3 times in a movie's overview, it has a high term frequency for that movie.
- **IDF (Inverse Document Frequency):** How rare is this word across ALL movies. The word
  "the" appears in almost every movie overview (low IDF = not distinctive). The word
  "lightsaber" appears in very few (high IDF = very distinctive).
- **TF-IDF = TF x IDF:** Words that are frequent in _this_ movie but rare across _all_
  movies get the highest scores. These are the words that make this movie unique.

**Why this matters for recommendations:** TF-IDF turns each movie into a vector (a list of
numbers), where each number represents how important a particular word is to that movie. We
can then compute the **cosine similarity** between two movies' vectors to get a number between
0 (completely different) and 1 (identical content).

### 4.2 The Name Cleaning Trick

This is the single most important concept in the feature engineering step.

```python
def _clean_name(name: str) -> str:
    """Remove spaces from a name to create a single TF-IDF token."""
    return str(name).lower().replace(' ', '')
```

**The problem:** TF-IDF splits text into individual words (called "tokens") using spaces as
separators. Consider the actor name "Johnny Depp":

```
Without cleaning:
  "Johnny Depp" ──> tokens: ["johnny", "depp"]

  Now imagine another movie has "Johnny Knoxville":
  "Johnny Knoxville" ──> tokens: ["johnny", "knoxville"]

  TF-IDF sees that both movies share the token "johnny" and considers
  them similar -- but Johnny Depp and Johnny Knoxville are completely
  different people! This is a FALSE match.
```

**The fix:** Remove spaces from names so the full name becomes a single token:

```
With cleaning:
  "Johnny Depp"      ──> token: ["johnnydepp"]
  "Johnny Knoxville" ──> token: ["johnnyknoxville"]

  Now TF-IDF sees these as two completely different tokens.
  No false match. Problem solved!
```

**Another example -- directors:**

```
Without cleaning:
  "Christopher Nolan" ──> ["christopher", "nolan"]
  "Christopher Columbus" ──> ["christopher", "columbus"]
  These would falsely appear similar because they share "christopher".

With cleaning:
  "Christopher Nolan"    ──> ["christophernolan"]
  "Christopher Columbus" ──> ["christophercolumbus"]
  Completely distinct tokens. Correct!
```

This same trick is also applied to **genres** and **keywords**:

```
Genre: "Science Fiction" ──> "sciencefiction"
  (Without this, "Science" alone could match "Science" from
   unrelated contexts)

Keyword: "martial arts" ──> "martialarts"
  (Prevents "arts" from matching "arts and crafts")
```

### 4.3 What Each Output Column Represents

The `build_features` function creates these cleaned text columns:

#### overview_clean

The movie's plot summary, lowercased. No special cleaning beyond lowercasing since the
overview is natural language text meant to be tokenized normally.

```
Before: "Led by Woody, Andy's toys live happily in his room until Andy's birthday..."
After:  "led by woody, andy's toys live happily in his room until andy's birthday..."
```

#### genres_str

All genres joined into a single string with spaces removed from each genre name.

```
Before (Python list): ["Animation", "Comedy", "Family"]
After (string):       "animation comedy family"

Before: ["Science Fiction", "Action", "Adventure"]
After:  "sciencefiction action adventure"
```

#### keywords_str

All keyword tags joined into a single string with spaces removed from multi-word keywords.

```
Before: ["jealousy", "toy", "boy", "friendship", "based on children's book"]
After:  "jealousy toy boy friendship basedonchildren'sbook"
```

#### cast_str

The top 5 cast member names, space-cleaned and joined.

```
Before: ["Tom Hanks", "Tim Allen", "Don Rickles", "Jim Varney", "Wallace Shawn"]
After:  "tomhanks timallen donrickles jimvarney wallaceshawn"
```

#### director_str

The director's name with spaces removed.

```
Before: "John Lasseter"
After:  "johnlasseter"
```

#### decade

Already formatted as a token from the earlier step. Passed through as-is.

```
"decade_1990s"
```

#### language

Already formatted as a token. Passed through as-is.

```
"lang_en"
```

#### collection

The franchise name with spaces removed and lowercased.

```
Before: "Toy Story Collection"
After:  "toystorycollection"
```

### 4.4 Complete Before/After Example

Let us trace a single movie -- **Toy Story (1995)** -- through the entire pipeline:

```
┌─────────────────────────── RAW DATA ───────────────────────────────┐
│                                                                     │
│  movies_metadata.csv:                                               │
│    id: 862                                                          │
│    title: "Toy Story"                                               │
│    overview: "Led by Woody, Andy's toys live happily in his         │
│              room, until Andy's birthday brings Buzz Lightyear..."  │
│    genres: '[{"id": 16, "name": "Animation"}, {"id": 35,           │
│             "name": "Comedy"}, {"id": 10751, "name": "Family"}]'   │
│    vote_average: 7.7                                                │
│    vote_count: 5415                                                 │
│    release_date: "1995-10-30"                                       │
│    original_language: "en"                                          │
│    belongs_to_collection: '{"name": "Toy Story Collection",...}'    │
│                                                                     │
│  credits.csv:                                                       │
│    cast: '[{"name":"Tom Hanks",...}, {"name":"Tim Allen",...},       │
│            {"name":"Don Rickles",...}, ...]'                         │
│    crew: '[{"name":"John Lasseter","job":"Director",...}, ...]'      │
│                                                                     │
│  keywords.csv:                                                      │
│    keywords: '[{"name":"jealousy"}, {"name":"toy"},                  │
│               {"name":"boy"}, {"name":"friendship"}, ...]'          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │  make_dataset.py
                              ▼
┌──────────────────── AFTER CLEANING & MERGING ──────────────────────┐
│                                                                     │
│  id: 862                                                            │
│  title: "Toy Story"                                                 │
│  overview: "Led by Woody, Andy's toys live happily..."              │
│  genres: ["Animation", "Comedy", "Family"]                          │
│  keywords: ["jealousy", "toy", "boy", "friendship"]                 │
│  cast: ["Tom Hanks", "Tim Allen", "Don Rickles",                    │
│         "Jim Varney", "Wallace Shawn"]                              │
│  director: "John Lasseter"                                          │
│  vote_average: 7.7                                                  │
│  vote_count: 5415                                                   │
│  decade: "decade_1990s"                                             │
│  language: "lang_en"                                                │
│  collection: "Toy Story Collection"                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │  build_features.py
                              ▼
┌──────────────────── AFTER FEATURE ENGINEERING ─────────────────────┐
│                                                                     │
│  overview_clean: "led by woody, andy's toys live happily..."        │
│  genres_str:     "animation comedy family"                          │
│  keywords_str:   "jealousy toy boy friendship"                      │
│  cast_str:       "tomhanks timallen donrickles jimvarney            │
│                   wallaceshawn"                                     │
│  director_str:   "johnlasseter"                                     │
│  decade:         "decade_1990s"                                     │
│  language:       "lang_en"                                          │
│  collection:     "toystorycollection"                               │
│                                                                     │
│  ──> All fields are now plain text strings ready for TF-IDF         │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.5 Why Separate Columns Instead of One Big String?

You might wonder: "Why not just mash everything together into one giant text field?"

The recommendation system uses **separate TF-IDF vectorizers** for each field, each with its
own **weight**. This lets us say "matching the same director is very important (weight: 2.0)
but matching the same decade is only slightly important (weight: 0.3)."

```
Field weights used by the recommender:
  overview:    1.0  (baseline)
  genres:      1.5  (important)
  keywords:    1.2  (moderately important)
  cast:        1.0  (baseline)
  director:    2.0  (very important -- same director is a strong signal)
  decade:      0.3  (minor boost)
  language:    0.5  (moderate)
  collection:  1.5  (important -- same franchise is very relevant)
```

If everything were in a single column, we could not control these weights -- "Christopher
Nolan" appearing in the director field would have the same importance as "Christopher Nolan"
appearing casually in an overview. Keeping them separate gives the system precise control over
what matters most.

---

## 5. Poster Fetching

File: **`src/data/fetch_posters.py`**

This script fetches movie poster image URLs from TMDB's website. These URLs are used purely
for the web interface (so users can see poster images while swiping through recommendations).
Posters are not part of the recommendation algorithm itself.

### 5.1 How It Works

1. **Load the processed movie list** and identify which movies do not yet have a poster URL
2. **For each movie**, construct its TMDB page URL (e.g., `https://www.themoviedb.org/movie/862`)
3. **Fetch the HTML** of that page using Python's built-in `urllib`
4. **Extract the poster URL** using a regular expression that finds image URLs in the page's
   structured data (JSON-LD metadata)
5. **Save the URL** to a cache file and eventually write all URLs back to the processed CSV

### 5.2 Rate Limiting and Politeness

TMDB is a real website. Hitting it with thousands of requests in rapid succession would
overload their servers and likely get our IP address blocked. The script includes several
safeguards:

- **Randomized delays (2-4 seconds)** between each request, so the traffic pattern looks more
  like a human browsing than a bot scraping
- **Rotating User-Agent headers** -- each request uses a randomly selected browser identity
  string (Chrome on Mac, Firefox on Windows, etc.) to avoid detection as a bot
- **Exponential backoff** on HTTP 429 ("Too Many Requests"): if the server says "slow down,"
  the script waits 60 seconds, then 120, then 240, up to a maximum of 600 seconds (10
  minutes)
- **Persistent cookie jar** -- maintains cookies across requests like a real browser would
- **Randomized request order** -- shuffles the list of movies to fetch so repeated runs do not
  hammer the same pages

### 5.3 Caching

Fetching ~9,000 poster URLs at 3 seconds each takes approximately **7.5 hours**. You do not
want to repeat that work. The script maintains a cache file (`data/processed/poster_cache.csv`)
that records which movies already have poster URLs. On subsequent runs, only movies not yet in
the cache are fetched.

The cache is also saved every 20 fetches during a run, so if the script crashes or is
interrupted, most progress is preserved.

```
First run:  0 cached  ──>  Fetch all 9,000  ──>  ~7.5 hours
Second run: 8,500 cached  ──>  Fetch remaining 500  ──>  ~25 min
Third run:  9,000 cached  ──>  "All done!" (instant)
```

---

## 6. Summary: The Complete Pipeline

Here is the entire data preparation process in one view:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: LOAD RAW DATA                                              │
│                                                                     │
│  5 CSV files from Kaggle ──> 5 pandas DataFrames in memory          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: CLEAN & FILTER                                             │
│                                                                     │
│  - Remove rows with corrupted non-numeric IDs                       │
│  - Keep only movies in the links_small subset (~9,000)              │
│  - Parse JSON strings into real Python objects (literal_eval)       │
│  - Extract top 5 cast, director, keywords, genres                   │
│  - Derive decade, language, collection tokens                       │
│  - Coerce numeric columns to float                                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: MERGE                                                      │
│                                                                     │
│  metadata + credits + keywords ──> single table (left joins on id)  │
│  Save to data/processed/movies_processed.csv                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: FEATURE ENGINEERING                                        │
│                                                                     │
│  For each field, create a clean text column:                        │
│  - overview_clean:  lowercased plot summary                         │
│  - genres_str:      "animation comedy family"                       │
│  - keywords_str:    "jealousy toy boy friendship"                   │
│  - cast_str:        "tomhanks timallen donrickles ..."              │
│  - director_str:    "johnlasseter"                                  │
│  - decade:          "decade_1990s"                                  │
│  - language:        "lang_en"                                       │
│  - collection:      "toystorycollection"                            │
│                                                                     │
│  Key trick: remove spaces from names so TF-IDF treats them as      │
│  single tokens (prevents false matches between unrelated people)    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: POSTER FETCHING (optional, for UI only)                    │
│                                                                     │
│  Scrape TMDB pages for poster image URLs                            │
│  Rate-limited, cached, takes ~7.5 hours for full dataset            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  READY FOR RECOMMENDATION                                           │
│                                                                     │
│  Each movie is now represented as clean text fields that TF-IDF     │
│  can convert into numerical vectors for similarity computation.     │
│                                                                     │
│  The recommender will:                                              │
│  1. Vectorize each field separately (TF-IDF or CountVectorizer)    │
│  2. Apply field-specific weights (director=2.0, genre=1.5, etc.)   │
│  3. Combine into one feature matrix                                 │
│  4. Compute cosine similarity between all pairs of movies           │
│  5. Use similarity scores + weighted ratings to rank recommendations│
└─────────────────────────────────────────────────────────────────────┘
```

### Key Takeaways

1. **Real-world data is messy.** IDs can be corrupted, JSON is stored as plain text strings,
   and you need to handle missing values at every step.

2. **Data from multiple sources must be merged carefully.** Left joins preserve all movies
   even when some metadata is missing, which is better than losing movies entirely.

3. **Feature engineering is where domain knowledge matters.** The name-cleaning trick (removing
   spaces) is not something a generic algorithm would figure out -- it requires understanding
   how TF-IDF tokenizes text and how that interacts with human names.

4. **Separate features enable weighted importance.** By keeping genres, cast, director, and
   other fields in separate columns, the system can weight them differently -- the director is
   worth 2x while the decade is worth 0.3x.

5. **External data (like posters) requires patience and politeness.** Fetching data from live
   websites means dealing with rate limits, caching, and potential failures.
