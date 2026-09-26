from typing import List
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
from normalizer import Normalizer
 
 
def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Word-level Jaccard similarity: |intersection| / |union| of the
    two strings' word sets. Range 0.0 - 1.0.
    """
    tokens_a = set(text_a.split(" ")) if text_a else set()
    tokens_b = set(text_b.split(" ")) if text_b else set()
 
    if not tokens_a and not tokens_b:
        return 1.0  # both empty -> treat as identical
    if not tokens_a or not tokens_b:
        return 0.0
 
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
 
 
def levenshtein_distance(a: str, b: str) -> int:
    """
    Classic edit-distance DP (pure Python, no external dependency).
    Number of single-character insertions/deletions/substitutions
    needed to turn `a` into `b`.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
 
    prev_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            curr_row[j] = min(
                prev_row[j] + 1,       # deletion
                curr_row[j - 1] + 1,   # insertion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = curr_row
 
    return prev_row[-1]
 
 
def levenshtein_ratio(text_a: str, text_b: str) -> float:
    """
    Normalized similarity from edit distance: 1.0 = identical,
    0.0 = completely different. This is the typo-tolerant metric --
    a one-character misspelling barely moves this score.
    """
    if not text_a and not text_b:
        return 1.0
 
    distance = levenshtein_distance(text_a, text_b)
    max_len = max(len(text_a), len(text_b))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)
 
 
class TfidfCosineScorer:
    """
    Character n-gram TF-IDF + cosine similarity.
 
    Must be fit once on the full corpus of normalized strings (all
    Source-1 + Source-2 + Source-3 names, or all addresses) so the
    vocabulary/IDF weights reflect the whole dataset -- not just the
    one pair being scored.
    """
 
    def __init__(self, ngram_range=(2, 3)):
        # analyzer="char_wb" -> character n-grams within word boundaries,
        # which is what makes this metric typo-tolerant: a misspelled
        # word still shares most of its n-grams with the correct one.
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)
        self._fitted = False
 
    def fit(self, corpus: List[str]) -> None:
        """corpus: list of normalized strings (e.g. every cleaned name)."""
        cleaned_corpus = [c if c else "" for c in corpus]
        self.vectorizer.fit(cleaned_corpus)
        self._fitted = True
 
    def score(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two normalized strings, 0.0 - 1.0."""
        if not self._fitted:
            raise RuntimeError("TfidfCosineScorer.fit(corpus) must be called before score().")
 
        vectors = self.vectorizer.transform([text_a or "", text_b or ""])
        sim = cosine_similarity(vectors[0], vectors[1])
        return float(sim[0][0])
 
 
class SimilarityFeatures:
    """
    Combines Jaccard, Levenshtein ratio, and TF-IDF cosine into one
    feature set for a (Source-1 text, candidate text) pair. Cleans
    both inputs with Normalizer.clean() before scoring.
 
    One TfidfCosineScorer should be fit on the full name corpus and
    another on the full address corpus, then passed in here.
    """
 
    def __init__(
        self,
        name_tfidf: TfidfCosineScorer,
        address_tfidf: TfidfCosineScorer,
        normalizer: Normalizer = None,
    ):
        self.name_tfidf = name_tfidf
        self.address_tfidf = address_tfidf
        self.normalizer = normalizer or Normalizer()
 
    def name_features(self, s1_name: str, other_name: str) -> dict:
        a = self.normalizer.clean(s1_name)
        b = self.normalizer.clean(other_name)
        return {
            "name_jaccard": jaccard_similarity(a, b),
            "name_levenshtein": levenshtein_ratio(a, b),
            "name_tfidf_cosine": self.name_tfidf.score(a, b),
        }
 
    def address_features(self, s1_address: str, other_address: str) -> dict:
        a = self.normalizer.clean(s1_address)
        b = self.normalizer.clean(other_address)
        return {
            "address_jaccard": jaccard_similarity(a, b),
            "address_levenshtein": levenshtein_ratio(a, b),
            "address_tfidf_cosine": self.address_tfidf.score(a, b),
        }
 
    def compute(self, s1_name, s1_address, other_name, other_address) -> dict:
        """All six features for one Source-1 / candidate pair."""
        features = {}
        features.update(self.name_features(s1_name, other_name))
        features.update(self.address_features(s1_address, other_address))
        return features