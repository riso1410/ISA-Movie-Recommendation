"""
Feature engineering: per-field text cleaning for multi-vectorizer recommender.
"""


def _clean_name(name: str) -> str:
    """Remove spaces from a name to create a single TF-IDF token."""
    return str(name).lower().replace(' ', '')


def _clean_name_list(names: list) -> str:
    """Clean a list of names and join with spaces."""
    if isinstance(names, list):
        return ' '.join(_clean_name(n) for n in names)
    return ''


def build_features(smd):
    """Create per-field cleaned text columns for multi-vectorizer pipeline."""
    smd = smd.copy()

    smd['overview_clean'] = smd['overview'].fillna('').astype(str).str.lower()
    smd['genres_str'] = smd['genres'].apply(
        lambda x: ' '.join(str(g).lower().replace(' ', '') for g in x) if isinstance(x, list) else ''
    )
    smd['keywords_str'] = smd['keywords'].apply(
        lambda x: ' '.join(str(k).lower().replace(' ', '') for k in x) if isinstance(x, list) else ''
    )
    smd['cast_str'] = smd['cast'].apply(_clean_name_list)
    smd['director_str'] = smd['director'].fillna('').apply(_clean_name)
    smd['decade'] = smd['decade'].fillna('') if 'decade' in smd.columns else ''
    smd['language'] = smd['language'].fillna('') if 'language' in smd.columns else ''
    smd['collection'] = smd['collection'].fillna('').apply(
        lambda x: str(x).lower().replace(' ', '') if x else ''
    ) if 'collection' in smd.columns else ''

    return smd
