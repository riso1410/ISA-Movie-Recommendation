"""
Feature engineering: text cleaning and metadata soup construction.
"""


def clean_text(text_list):
    """Remove spaces from names to create single tokens."""
    if isinstance(text_list, list):
        return [str(x).lower().replace(' ', '') for x in text_list]
    return str(text_list).lower().replace(' ', '')


def create_soup(row):
    """Combine all metadata features into a single string for TF-IDF."""
    genres = ' '.join(clean_text(row['genres']))
    keywords = ' '.join(clean_text(row['keywords']))
    cast = ' '.join(clean_text(row['cast']))
    director = ' '.join([clean_text(row['director'])] * 3)
    overview = str(row['overview']).lower()
    return f"{overview} {genres} {keywords} {cast} {director}"


def build_features(smd):
    """Add soup column to the dataframe."""
    smd = smd.copy()
    smd['soup'] = smd.apply(create_soup, axis=1)
    return smd
