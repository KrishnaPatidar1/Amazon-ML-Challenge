"""
normalizer.py
 
Cleans a business_name or business_address string into a standard form:
1. lowercase
2. remove extra/multiple spaces
3. expand short forms / abbreviations into full forms
4. remove filler words (the, and, &, etc.)
5. return the cleaned string
"""
 
import re
 
# Short-form -> full-form expansions.
# Covers both name (legal suffix) and address abbreviations, since the
# same "expand short forms" step applies to either kind of input.
SHORT_FORMS = {
    # business/legal suffixes
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "co": "company",
 
    # address abbreviations
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "apt": "apartment",
    "flr": "floor",
    "bldg": "building",
    "ste": "suite",
    "hwy": "highway",
    "no": "number",
    "opp": "opposite",
    "nr": "near",
}
 
# Filler words to drop entirely (not expanded, just removed).
FILLER_WORDS = {"the", "and", "&"}
 
 
class Normalizer:
    """Cleans business name / address strings into a standard format."""
 
    def __init__(self, short_forms: dict = None, filler_words: set = None):
        self.short_forms = short_forms or SHORT_FORMS
        self.filler_words = filler_words or FILLER_WORDS
 
    def clean(self, text: str) -> str:
        """
        Take a name or address string and return the cleaned version:
        lowercase, extra spaces removed, short forms expanded, filler
        words removed.
        """
        # Treat both Python None and pandas/numpy NaN (a float) as missing.
        # A raw `is None` check misses NaN, which would otherwise get
        # stringified into the literal text "nan" and matched as real data.
        if text is None or (isinstance(text, float) and text != text):
            return ""
 
        # 1. lowercase
        text = str(text).lower()
 
        # remove punctuation so words split cleanly (kept as part of
        # "remove extra spaces / clean up" step)
        text = re.sub(r"[^a-z0-9\s&]", " ", text)
 
        # 2. remove extra/multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
 
        # 3. expand short forms + 4. remove filler words
        cleaned_tokens = []
        for token in text.split():
            if token in self.filler_words:
                continue
            token = self.short_forms.get(token, token)
            cleaned_tokens.append(token)
 
        # 2 (again). collapse spaces after token changes
        cleaned = " ".join(cleaned_tokens)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
 
        # 5. return the cleaned string
        return cleaned
 
    def normalize_name(self, name: str) -> str:
        return self.clean(name)
 
    def normalize_address(self, address: str) -> str:
        return self.clean(address)