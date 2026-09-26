"""
blocker.py
 
match(): given one Source-1 record's name and address, checks it against
every row of source2_df and source3_df. Normalizes each name/address
first, then scores similarity using Jaccard (word overlap) and
Levenshtein ratio (typo-tolerant, character-level). If either score
clears its threshold, that row's entity_id is added to the result list.
"""
 
from normalizer import Normalizer
from sim_features import jaccard_similarity, levenshtein_ratio
 
normalizer = Normalizer()
 
 
def _field_score(text_a: str, text_b: str) -> float:
    """Combine word-level and character-level similarity for one field."""
    return max(jaccard_similarity(text_a, text_b), levenshtein_ratio(text_a, text_b))
 
 
def match(
    s1_name: str,
    s1_address: str,
    source2_df,
    source3_df,
    name_threshold: float = 0.85,
    address_threshold: float = 0.85,
) -> list:
    """
    s1_name, s1_address: raw (un-normalized) name/address of one Source-1 entity.
    source2_df, source3_df: pandas DataFrames with columns
        entity_id, business_name, business_address (NaNs should already
        be filled with "" before calling this).
 
    Returns: list of entity_ids from source2_df/source3_df that matched.
    """
    clean_s1_name = normalizer.normalize_name(s1_name)
    clean_s1_address = normalizer.normalize_address(s1_address)
 
    matched_ids = []
 
    for df in (source2_df, source3_df):
        for _, row in df.iterrows():
            other_name = normalizer.normalize_name(row["business_name"])
            other_address = normalizer.normalize_address(row["business_address"])
            
            name_score = _field_score(clean_s1_name, other_name)
            address_score = _field_score(clean_s1_address, other_address)
 
            if name_score >= name_threshold or address_score >= address_threshold:
                matched_ids.append(row["entity_id"])
 
    return matched_ids