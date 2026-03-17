"""
Data loading and cleaning pipeline.
Loads raw CSVs, cleans IDs, merges datasets, and saves processed data.
"""

import pandas as pd
import numpy as np
from ast import literal_eval
from pathlib import Path


TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


def load_raw_data(raw_dir="data/raw"):
    """Load all raw CSV files."""
    raw_dir = Path(raw_dir)
    metadata = pd.read_csv(raw_dir / "movies_metadata.csv", low_memory=False)
    credits = pd.read_csv(raw_dir / "credits.csv")
    keywords = pd.read_csv(raw_dir / "keywords.csv")
    links_small = pd.read_csv(raw_dir / "links_small.csv")
    return metadata, credits, keywords, links_small


def clean_metadata(metadata, links_small):
    """Clean metadata: fix IDs, filter to small subset."""
    metadata = metadata[metadata["id"].apply(lambda x: str(x).isdigit())].copy()
    metadata["id"] = metadata["id"].astype(int)

    links_small = links_small[links_small["tmdbId"].notna()].copy()
    links_small["tmdbId"] = links_small["tmdbId"].astype(int)
    smd = metadata[metadata["id"].isin(links_small["tmdbId"])].copy()

    return smd


def parse_credits(credits_df):
    """Parse cast (top 5) and director from credits."""
    cred = credits_df.copy()
    cred["id"] = cred["id"].astype(int)
    cred["cast"] = cred["cast"].apply(
        lambda x: [d["name"] for d in literal_eval(x)][:5] if isinstance(x, str) else []
    )
    cred["director"] = cred["crew"].apply(
        lambda x: next(
            (d["name"] for d in literal_eval(x) if d["job"] == "Director"), np.nan
        )
        if isinstance(x, str)
        else np.nan
    )
    return cred[["id", "cast", "director"]]


def parse_keywords(keywords_df):
    """Parse keyword names from keywords."""
    kw = keywords_df.copy()
    kw["id"] = kw["id"].astype(int)
    kw["keywords"] = kw["keywords"].apply(
        lambda x: [d["name"] for d in literal_eval(x)] if isinstance(x, str) else []
    )
    return kw[["id", "keywords"]]


def parse_decade(date_str: str) -> str:
    """Extract decade token from release_date string, e.g. 'decade_1990s'."""
    try:
        year = int(str(date_str)[:4])
        decade = (year // 10) * 10
        return f"decade_{decade}s"
    except (ValueError, TypeError):
        return ""


def parse_language(lang: str) -> str:
    """Convert language code to token, e.g. 'lang_en'."""
    if pd.isna(lang) or not str(lang).strip():
        return ""
    return f"lang_{str(lang).strip().lower()}"


def parse_collection(collection_str: str) -> str:
    """Extract collection name from belongs_to_collection JSON string."""
    if pd.isna(collection_str) or not str(collection_str).strip():
        return ""
    try:
        col = literal_eval(str(collection_str))
        if isinstance(col, dict) and "name" in col:
            return col["name"]
    except (ValueError, SyntaxError):
        pass
    return ""


def build_poster_url(poster_path: str) -> str:
    """Convert TMDB poster_path to a full image URL."""
    if pd.isna(poster_path):
        return ""

    poster_path = str(poster_path).strip()
    if not poster_path or poster_path.lower() == "nan":
        return ""

    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"


def merge_datasets(smd, credits_parsed, keywords_parsed):
    """Merge metadata with parsed credits and keywords."""
    smd["genres"] = (
        smd["genres"]
        .fillna("[]")
        .apply(literal_eval)
        .apply(lambda x: [d["name"] for d in x] if isinstance(x, list) else [])
    )
    smd = smd.merge(keywords_parsed, on="id", how="left")
    smd["keywords"] = (
        smd["keywords"].fillna("").apply(lambda x: x if isinstance(x, list) else [])
    )
    smd = smd.merge(credits_parsed, on="id", how="left")
    smd["cast"] = (
        smd["cast"].fillna("").apply(lambda x: x if isinstance(x, list) else [])
    )
    smd["director"] = smd["director"].fillna("")
    smd["overview"] = smd["overview"].fillna("")

    for col in ["vote_average", "vote_count", "popularity"]:
        smd[col] = pd.to_numeric(smd[col], errors="coerce").fillna(0.0)

    smd["decade"] = smd["release_date"].apply(parse_decade)
    smd["language"] = smd["original_language"].apply(parse_language)
    smd["collection"] = smd["belongs_to_collection"].apply(parse_collection)

    smd = smd[smd["title"].notna()]
    return smd


def make_dataset(raw_dir="data/raw", processed_dir="data/processed"):
    """Full pipeline: load, clean, parse, merge, save."""
    metadata, credits, keywords, links_small = load_raw_data(raw_dir)
    smd = clean_metadata(metadata, links_small)
    credits_parsed = parse_credits(credits)
    keywords_parsed = parse_keywords(keywords)
    smd = merge_datasets(smd, credits_parsed, keywords_parsed)
    smd["poster_url"] = (
        smd["poster_path"].apply(build_poster_url)
        if "poster_path" in smd.columns
        else ""
    )

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "director",
        "vote_average",
        "vote_count",
        "popularity",
        "decade",
        "language",
        "collection",
        "poster_url",
    ]

    out_path = processed_dir / "movies_processed.csv"
    smd[cols].to_csv(out_path, index=False)
    print(f"Saved {len(smd)} movies to {processed_dir / 'movies_processed.csv'}")
    return smd


if __name__ == "__main__":
    make_dataset()
