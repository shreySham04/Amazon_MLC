"""
normalize.py
============
Country-agnostic normalisation of business names and addresses.
Handles: English, Hindi (transliterations), French, and transliterations.

Key outputs per record:
  - name_norm          : cleaned, lowercased name
  - name_core          : name without legal suffixes / stopwords
  - name_tokens_sorted : sorted token bag (order-invariant matching)
  - name_acronym       : initialism from core tokens
  - addr_norm          : cleaned address
  - addr_tokens        : cleaned address token set
  - postal_code        : extracted postal/PIN/ZIP code
  - street_num         : street/house number
  - city               : extracted city token
"""

import re
import unicodedata
from typing import Optional

# ─── Legal suffix expansions (name → canonical) ─────────────────────────────
LEGAL_SUFFIX_MAP = {
    # English
    r"\bllc\b": "llc",
    r"\bl\.l\.c\.?\b": "llc",
    r"\binc\.?\b": "inc",
    r"\bincorporated\b": "inc",
    r"\bltd\.?\b": "ltd",
    r"\blimited\b": "ltd",
    r"\bcorp\.?\b": "corp",
    r"\bcorporation\b": "corp",
    r"\bco\.?\b": "co",
    r"\bcompany\b": "co",
    r"\bpvt\.?\b": "pvt",
    r"\bprivate\b": "pvt",
    r"\bllp\b": "llp",
    r"\blimited liability partnership\b": "llp",
    r"\bplc\b": "plc",
    r"\b(pvt\.?\s*ltd\.?|private limited)\b": "pvt ltd",
    r"\bsdn\.?\s*bhd\.?\b": "sdn bhd",
    r"\bpt\.?\b": "pt",
    # French
    r"\bsarl\b": "sarl",
    r"\bsas\b": "sas",
    r"\bsa\b": "sa",
    r"\beurl\b": "eurl",
    r"\bsci\b": "sci",
    r"\bsnc\b": "snc",
    # German/Other common
    r"\bgmbh\b": "gmbh",
    r"\bag\b": "ag",
    r"\bab\b": "ab",
    r"\bbv\b": "bv",
    r"\bnv\b": "nv",
    # India
    r"\bpvt\.\s*ltd\.?\b": "pvt ltd",
}

LEGAL_SUFFIX_SET = {
    "llc", "inc", "ltd", "corp", "co", "pvt", "llp", "plc",
    "sarl", "sas", "sa", "eurl", "sci", "snc", "gmbh", "ag",
    "ab", "bv", "nv", "pvt ltd", "sdn bhd", "pt", "lp",
}

# ─── Address abbreviation expansions ────────────────────────────────────────
ADDR_ABBREV = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bdr\.?\b": "drive",
    r"\bln\.?\b": "lane",
    r"\bct\.?\b": "court",
    r"\bpl\.?\b": "place",
    r"\bsq\.?\b": "square",
    r"\bhwy\.?\b": "highway",
    r"\bfwy\.?\b": "freeway",
    r"\bpkwy\.?\b": "parkway",
    r"\bexpy\.?\b": "expressway",
    r"\bste\.?\b": "suite",
    r"\bapt\.?\b": "apartment",
    r"\bflr\.?\b": "floor",
    r"\bbldg\.?\b": "building",
    # French
    r"\bbd\.?\b": "boulevard",
    r"\brue\b": "rue",
    r"\bav\.?\b": "avenue",
    r"\bres\.?\b": "residence",
    r"\bbt\.?\b": "batiment",
    # Indian
    r"\bph\.?\b": "phase",
    r"\bsec\.?\b": "sector",
    r"\bno\.?\b": "number",
    r"\bnear\b": "near",
}

# Tokens that are noise in address (landmarks etc.) – down-weight, don't remove
LANDMARK_TOKENS = {
    "near", "opposite", "opp", "behind", "above", "below",
    "beside", "adjacent", "next", "landmark",
}

# ─── Compile patterns ────────────────────────────────────────────────────────
_LEGAL_PATTERNS = [(re.compile(pat, re.IGNORECASE), repl)
                   for pat, repl in LEGAL_SUFFIX_MAP.items()]
_ADDR_PATTERNS = [(re.compile(pat, re.IGNORECASE), repl)
                  for pat, repl in ADDR_ABBREV.items()]

# Postal/PIN/ZIP detector: 5-6 digit US ZIP, Indian PIN, French postal code,
# also UK-style (not critical), and alphanumeric combos
_POSTAL_RE = re.compile(
    r"\b(\d{5,6}(?:-\d{4})?)\b",  # US ZIP 12345 or 12345-6789, or Indian PIN 6-digit
)
# Street/house number (at start of address or as a component)
_STREET_NUM_RE = re.compile(r"^\s*(\d+\s*(?:[a-z]?\b|/\d+)?)", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")  # keep word chars and spaces


def _unicode_normalize(text: str) -> str:
    """NFKD decompose, strip combining marks (accents), re-encode to ASCII-safe."""
    text = unicodedata.normalize("NFKD", text)
    # Strip combining diacritical marks
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def _amp_expand(text: str) -> str:
    """Replace & and + with 'and'."""
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"\s*\+\s*", " and ", text)
    return text


def normalize_name(raw: Optional[str]) -> dict:
    """
    Normalise a business name. Returns a dict with keys:
      name_norm, name_core, name_tokens_sorted, name_acronym
    """
    if not raw or not isinstance(raw, str) or raw.strip() == "":
        return {
            "name_norm": "",
            "name_core": "",
            "name_tokens_sorted": "",
            "name_acronym": "",
        }

    text = str(raw)
    # Unicode normalise + accent strip
    text = _unicode_normalize(text)
    text = text.lower()
    # & / + → and
    text = _amp_expand(text)
    # Strip leading punctuation noise (e.g. "-- Holloway Peak Inc")
    text = re.sub(r"^[\-\.\,\s]+", "", text)
    # Remove extraneous punctuation (keep hyphens in compound words)
    text = re.sub(r"[^\w\s\-]", " ", text)
    # Apply legal suffix canonicalisation
    for pat, repl in _LEGAL_PATTERNS:
        text = pat.sub(repl, text)
    # Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    name_norm = text

    # Core name: remove legal suffix tokens + generic stopwords
    stopwords = {"the", "a", "an", "of", "and", "or", "for", "in", "at", "by"}
    tokens = name_norm.split()
    core_tokens = [t for t in tokens
                   if t not in LEGAL_SUFFIX_SET and t not in stopwords]
    name_core = " ".join(core_tokens)
    name_tokens_sorted = " ".join(sorted(core_tokens))

    # Acronym from first letters of core tokens (≥2 chars each)
    name_acronym = "".join(t[0] for t in core_tokens if len(t) >= 2)

    return {
        "name_norm": name_norm,
        "name_core": name_core,
        "name_tokens_sorted": name_tokens_sorted,
        "name_acronym": name_acronym,
    }


def normalize_address(raw: Optional[str]) -> dict:
    """
    Normalise a business address. Returns a dict with keys:
      addr_norm, addr_tokens, postal_code, street_num, has_landmark
    """
    if not raw or not isinstance(raw, str) or raw.strip() == "":
        return {
            "addr_norm": "",
            "addr_tokens": "",
            "postal_code": "",
            "street_num": "",
            "has_landmark": 0,
        }

    text = str(raw)
    text = _unicode_normalize(text)
    text = text.lower()
    text = _amp_expand(text)

    # Extract postal code before stripping digits
    postal_match = _POSTAL_RE.search(text)
    postal_code = postal_match.group(1).replace("-", "") if postal_match else ""

    # Extract leading street number
    snum_match = _STREET_NUM_RE.match(text)
    street_num = snum_match.group(1).strip() if snum_match else ""

    # Apply address abbreviation expansion
    for pat, repl in _ADDR_PATTERNS:
        text = pat.sub(repl, text)

    # Remove punctuation (keep commas as token separators, replace with space)
    text = re.sub(r"[,/\-\.]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    addr_norm = text

    # Token set for overlap features
    tokens = set(addr_norm.split())
    # Check for landmark tokens
    has_landmark = int(bool(tokens & LANDMARK_TOKENS))
    # Remove very short tokens (noise) and digits-only short tokens
    tokens = {t for t in tokens if len(t) >= 2}
    addr_tokens = " ".join(sorted(tokens))

    return {
        "addr_norm": addr_norm,
        "addr_tokens": addr_tokens,
        "postal_code": postal_code,
        "street_num": street_num,
        "has_landmark": has_landmark,
    }


def normalize_record(entity_id: str, name: Optional[str],
                     address: Optional[str], country: Optional[str]) -> dict:
    """Normalise a single record. Returns flat dict of all features."""
    rec = {"entity_id": entity_id, "country": (country or "").strip().lower()}
    rec.update(normalize_name(name))
    rec.update(normalize_address(address))
    return rec
