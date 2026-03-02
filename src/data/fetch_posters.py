"""
Fetch current poster URLs by scraping TMDB movie pages.
Sequential requests with randomized delays and rotating headers.

Usage:
    uv run python -m src.data.fetch_posters
"""
import random
import re
import time
import urllib.request
import urllib.error
from http.cookiejar import CookieJar
from pathlib import Path

import pandas as pd

TMDB_URL = "https://www.themoviedb.org/movie/{movie_id}"
IMAGE_RE = re.compile(r'"image"\s*:\s*"(https://image\.tmdb\.org/t/p/w500/[^"]+)"')

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:133.0) Gecko/20100101 Firefox/133.0",
]

DELAY_MIN = 2.0       # min seconds between requests
DELAY_MAX = 4.0       # max seconds between requests
BACKOFF_BASE = 60     # initial backoff on 429
BACKOFF_MAX = 600     # max backoff
TIMEOUT = 15          # request timeout
SAVE_EVERY = 20       # save cache after this many fetches

CACHE_PATH = Path("data/processed/poster_cache.csv")
PROCESSED_PATH = Path("data/processed/movies_processed.csv")

# Persistent cookie jar + opener
_cookie_jar = CookieJar()
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_cookie_jar))


def _random_headers() -> dict[str, str]:
    """Generate randomized browser-like headers."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity",
        "Referer": "https://www.themoviedb.org/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def fetch_one(movie_id: int) -> str | None:
    """Fetch poster URL for one movie. Returns URL, empty string (no poster), or None (error/rate-limited)."""
    url = TMDB_URL.format(movie_id=movie_id)
    try:
        req = urllib.request.Request(url, headers=_random_headers())
        with _opener.open(req, timeout=TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        match = IMAGE_RE.search(html)
        return match.group(1) if match else ""
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return None  # rate limited
        if e.code == 404:
            return ""    # movie doesn't exist
        return None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def load_cache() -> dict[int, str]:
    """Load successful poster URLs from cache."""
    if not CACHE_PATH.exists():
        return {}
    df = pd.read_csv(CACHE_PATH)
    df = df[df["poster_url"].notna() & (df["poster_url"] != "")]
    return dict(zip(df["id"].astype(int), df["poster_url"]))


def save_cache(results: dict[int, str]) -> None:
    """Save poster URLs to cache."""
    df = pd.DataFrame(list(results.items()), columns=["id", "poster_url"])
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)


def main() -> None:
    df = pd.read_csv(PROCESSED_PATH)
    all_ids = df["id"].astype(int).tolist()

    cache = load_cache()
    print(f"Total movies: {len(all_ids)}")
    print(f"Already cached: {len(cache)}")

    remaining = [mid for mid in all_ids if mid not in cache]
    random.shuffle(remaining)  # randomize access order
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        _write_to_csv(df, cache)
        return

    avg_delay = (DELAY_MIN + DELAY_MAX) / 2
    est_min = len(remaining) * avg_delay / 60
    print(f"Estimated time: ~{est_min:.0f} min ({avg_delay:.1f}s avg/req)")
    print()

    backoff = 0
    fetched = 0
    found = 0
    skipped = 0
    consecutive_fails = 0

    for i, mid in enumerate(remaining):
        result = fetch_one(mid)
        status = "✓" if result else ("⨯" if result == "" else "⏳ 429")
        print(f"  [{i+1}/{len(remaining)}] id={mid} {status}", flush=True)

        if result is None:
            consecutive_fails += 1
            backoff = min(max(backoff * 2, BACKOFF_BASE), BACKOFF_MAX)
            print(f"  [{i+1}/{len(remaining)}] id={mid} — rate limited, "
                  f"backing off {backoff}s...")
            save_cache(cache)
            time.sleep(backoff)
            # Retry once
            result = fetch_one(mid)
            if result is None:
                skipped += 1
                if consecutive_fails >= 5:
                    print(f"  5 consecutive failures — pausing 10min...")
                    time.sleep(600)
                    consecutive_fails = 0
                continue
            backoff = 0
            consecutive_fails = 0
        else:
            backoff = 0
            consecutive_fails = 0

        if result:
            cache[mid] = result
            found += 1
        else:
            skipped += 1

        fetched += 1

        if fetched % SAVE_EVERY == 0:
            save_cache(cache)
            pct = (i + 1) / len(remaining) * 100
            print(f"  [{i+1}/{len(remaining)}] {pct:.1f}% — "
                  f"{found} posters, {skipped} missing")

        # Randomized delay
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    save_cache(cache)
    print(f"\nFetch complete: {found} new posters, {skipped} missing")
    print(f"Total cached: {len(cache)}/{len(all_ids)}")

    _write_to_csv(df, cache)


def _write_to_csv(df: pd.DataFrame, cache: dict[int, str]) -> None:
    """Write poster_url column to processed CSV."""
    df["poster_url"] = df["id"].map(cache).fillna("")
    df.to_csv(PROCESSED_PATH, index=False)
    has = (df["poster_url"] != "").sum()
    print(f"Saved {PROCESSED_PATH}: {has}/{len(df)} have poster URLs")


if __name__ == "__main__":
    main()
