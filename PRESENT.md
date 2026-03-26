# Presentation Notes — `feature/topk-5` branch

> Personal cheat-sheet: what changed and why. You didn't write this code — Rišo did.

---

## 1. Multi-Model Architecture (the big change)

**Before:** Single recommender model (`recommender.pkl`) using the multi-vectorizer approach only.

**Now:** 3 separate models, each a different recommendation strategy:

| Model      | How it works                                                                                             | File                       |
| ---------- | -------------------------------------------------------------------------------------------------------- | -------------------------- |
| `overview` | TF-IDF only on movie plot text (baseline)                                                                | `recommender_overview.pkl` |
| `soup`     | TF-IDF on concatenated metadata (overview + genres + keywords + cast + director×3)                       | `recommender_soup.pkl`     |
| `multi`    | Per-field vectorizers with separate weights (overview=1.0, director=2.0, genres=1.5, etc.) — **default** | `recommender_multi.pkl`    |

**Why this matters:** You can now compare how different feature strategies affect recommendations. Overview-only is the simplest baseline, soup is a common approach from tutorials, multi is the most sophisticated.

**Where to show:** `src/models/predict_model.py` — the `ContentBasedRecommender` class now takes a `mode` param. `fit()` dispatches to `_fit_overview()`, `_fit_soup()`, or `_fit_multi()`.

**Training:** `train_model.py` loops over all 3 modes and saves each as `recommender_<mode>.pkl`.

**Simple analogy — what does each model use to decide "similar movies"?**

| Model        | What it looks at                                                                    | Analogy                                                                                           |
| ------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **overview** | Only the plot description text                                                      | "Recommend me movies with a similar story"                                                        |
| **soup**     | Everything mashed into one text blob (plot + genres + keywords + cast + director×3) | "Throw all info into a blender, find similar smoothies"                                           |
| **multi**    | Each field separately with different importance weights                             | "Director matters 2× more than cast, genres matter 1.5×" — like a judge scoring multiple criteria |

`overview` and `soup` treat all words equally in one big bag. `multi` gives each field its own vectorizer and weight. `soup`'s director×3 trick repeats the director name so TF-IDF naturally gives it more weight — a poor man's version of what `multi` does with explicit weights.

---

## 2. Model Switching in the App

**New UI element:** Dropdown in the header lets you switch models live.

**Backend:** `POST /api/models/switch` swaps the active recommender. `GET /api/models` lists available models. On startup, all `recommender_*.pkl` files from `models/` are loaded into memory.

**What to demo:** Start the app, swipe a few movies, then switch from `multi` to `overview` — the recommendations change because the similarity matrix is different.

---

## 3. Soup Column in Feature Engineering

`build_features.py` now also builds a `soup` column — concatenation of all text fields with director repeated 3× for emphasis. This feeds the `soup` mode.

---

## 4. Poster URLs (inline, no external fetching)

**Before:** `fetch_posters.py` hit the TMDB API for each movie's poster URL.

**Now:** `make_dataset.py` builds poster URLs directly from the `poster_path` column already in the metadata CSV: `https://image.tmdb.org/t/p/w500{poster_path}`. Covers 9216/9219 movies (99.97%). The `fetch_posters.py` script was deleted.

---

## 5. Metrics Standardized to @5

All evaluation metrics changed from @10 to **@5** (Precision@5, NDCG@5, MAP@5, Coverage@5, etc.). Matches the "top-K=5" focus of the branch name.

New metrics added: **MAP@5** (Mean Average Precision) and **MRR** (Mean Reciprocal Rank).

All metrics use **genre overlap** as the definition of "relevant" — if a recommended movie shares at least 1 genre with the input movie, it counts as a hit.

| Metric          | What it measures                                            | Simple example (K=5)                                  |
| --------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| **Precision@5** | Out of 5 recs, how many share a genre?                      | 4/5 share a genre → 0.80                              |
| **NDCG@5**      | Are the _best_ matches ranked first?                        | Most relevant rec at #1 → high. At #5 → low.          |
| **MAP@5**       | Like Precision but rewards finding relevant items _earlier_ | Hits at positions 1,3 → better than hits at 4,5       |
| **MRR**         | How quickly do you find the _first_ relevant item?          | First hit at #1 → 1.0. First hit at #3 → 0.33         |
| **Coverage**    | What % of catalog ever gets recommended?                    | 500 out of 9219 movies appear → 5.4%                  |
| **Diversity**   | How different are the 5 recs from each other?               | 5 unique genres = high. 5 nearly identical = low.     |
| **Novelty**     | Does it recommend obscure or popular movies?                | Unknown gems = high novelty. Only blockbusters = low. |

**The @5 part** = "look at top 5 recommendations." Previously @10. Smaller K is stricter — fewer chances to score well.

---

## 6. Frontend Cleanup

- **Removed:** Separate dashboard page (`dashboard.html` deleted — was 3572 lines)
- **Added:** Single "Analysis" tab inside the main SPA showing model stats, metrics, per-genre precision
- **Tab navigation** instead of separate pages
- Metrics display cleaned to show @5 values

---

## 7. Massive Cleanup (~11,600 lines deleted)

Deleted files:

- `docs/` — Sphinx config, old plans, METRICS.md, evaluation report
- `learn/` — 7 educational markdown docs (00-06)
- `app/static/dashboard.html` — old separate analytics page
- `src/data/fetch_posters.py` — replaced by inline URL building
- `src/visualization/` — empty placeholder files

**Why:** These were scaffolding/docs not needed for the final deliverable.

---

## 8. Backend Simplification

- Removed background task infrastructure (threading, task queue, progress tracking)
- Removed `/api/weights` endpoint (live weight tuning) — weights are now baked into each model
- Cleaner startup: loads all `.pkl` files from `models/`, picks `multi` as default
- `_prepare_model()` helper handles poster URL mapping consistently

---

## Demo Flow

1. Start app: `uv run uvicorn app.main:app --reload`
2. Show the swipe UI working with poster images
3. Swipe 3+ right to trigger content-based recs
4. Switch model via dropdown → show recs change
5. Open Analysis tab → show metrics at @5
