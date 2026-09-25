"""
features.py
===========
Feature engineering for matched (S1, candidate) pairs.

Feature groups:
  1. Name features       – string similarity on name_norm, name_core, name_tokens_sorted
  2. Address features    – postal code, street number, token overlap, address similarity
  3. Semantic features   – multilingual embedding cosine similarity
  4. Context features    – candidate rank, score gap, source (S2 vs S3)
  5. Meta features       – missing-field indicators
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure the src/ directory is on the path so sibling modules resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from rapidfuzz import distance as rfd
from rapidfuzz import fuzz
from normalize import LEGAL_SUFFIX_SET

logger = logging.getLogger(__name__)


# ─── String similarity helpers ───────────────────────────────────────────────

def _safe_jaro_winkler(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def _safe_jw(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return rfd.JaroWinkler.normalized_similarity(a, b)


def _jaccard_tokens(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _jaccard_chars(a: str, b: str, n: int = 3) -> float:
    def ngrams(s, n):
        return set(s[i:i+n] for i in range(len(s)-n+1))
    sa = ngrams(a, n)
    sb = ngrams(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _normalized_edit(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return rfd.Levenshtein.normalized_similarity(a, b)


def _partial_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.partial_ratio(a, b) / 100.0


def _token_sort_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def _token_set_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def _acronym_match(a: str, b: str) -> int:
    """1 if one is an acronym of the other."""
    if not a or not b:
        return 0
    return int(a == b or a.startswith(b) or b.startswith(a))


def _legal_suffix_agreement(name_a: str, name_b: str) -> float:
    """Fraction of legal suffix tokens shared."""
    def suffixes(name):
        return {t for t in name.split() if t in LEGAL_SUFFIX_SET}
    sa = suffixes(name_a)
    sb = suffixes(name_b)
    if not sa and not sb:
        return 1.0  # both have no suffix → agree
    if not sa or not sb:
        return 0.5  # one has, other doesn't → partial
    return len(sa & sb) / len(sa | sb)


def _exact_match(a: str, b: str) -> int:
    if not a or not b:
        return 0
    return int(a == b)


# ─── Address helpers ─────────────────────────────────────────────────────────

def _postal_match(a: str, b: str) -> float:
    """0.0 missing, 0.5 prefix match, 1.0 exact."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a[:3] == b[:3]:  # 3-digit prefix match
        return 0.5
    return 0.0


def _street_num_match(a: str, b: str) -> float:
    """1.0 exact, 0.5 partial overlap, 0.0 no match or missing."""
    if not a or not b:
        return 0.0
    a_num = "".join(c for c in a if c.isdigit())
    b_num = "".join(c for c in b if c.isdigit())
    if not a_num or not b_num:
        return 0.0
    return 1.0 if a_num == b_num else 0.0


def _addr_token_overlap(a_tokens: str, b_tokens: str) -> float:
    """Jaccard over address token sets."""
    return _jaccard_tokens(a_tokens, b_tokens)


# ─── Feature extraction for a pair ───────────────────────────────────────────

def compute_pair_features(row_s1: dict, row_cand: dict,
                          s1_emb: Optional[np.ndarray] = None,
                          cand_emb: Optional[np.ndarray] = None) -> dict:
    """
    Compute all features for a single (S1, candidate) pair.

    Parameters
    ----------
    row_s1   : dict of normalised S1 fields
    row_cand : dict of normalised candidate fields
    s1_emb   : optional embedding vector for S1 record
    cand_emb : optional embedding vector for candidate record
    """
    feats = {}

    n1 = row_s1.get("name_norm", "") or ""
    n2 = row_cand.get("name_norm", "") or ""
    c1 = row_s1.get("name_core", "") or ""
    c2 = row_cand.get("name_core", "") or ""
    ts1 = row_s1.get("name_tokens_sorted", "") or ""
    ts2 = row_cand.get("name_tokens_sorted", "") or ""
    ac1 = row_s1.get("name_acronym", "") or ""
    ac2 = row_cand.get("name_acronym", "") or ""

    # ── Name features ──────────────────────────────────────────────────────
    feats["name_jaccard_tokens"] = _jaccard_tokens(n1, n2)
    feats["name_jaccard_chars3"] = _jaccard_chars(n1, n2, 3)
    feats["name_jaccard_chars4"] = _jaccard_chars(n1, n2, 4)
    feats["name_edit_dist"] = _normalized_edit(n1, n2)
    feats["name_jaro_winkler"] = _safe_jw(n1, n2)
    feats["name_partial_ratio"] = _partial_ratio(n1, n2)
    feats["name_token_sort_ratio"] = _token_sort_ratio(n1, n2)
    feats["name_token_set_ratio"] = _token_set_ratio(n1, n2)

    feats["core_jaccard_tokens"] = _jaccard_tokens(c1, c2)
    feats["core_edit_dist"] = _normalized_edit(c1, c2)
    feats["core_jaro_winkler"] = _safe_jw(c1, c2)
    feats["core_partial_ratio"] = _partial_ratio(c1, c2)
    feats["core_token_sort_ratio"] = _token_sort_ratio(c1, c2)
    feats["core_token_set_ratio"] = _token_set_ratio(c1, c2)

    feats["sorted_tokens_exact"] = _exact_match(ts1, ts2)
    feats["sorted_tokens_edit"] = _normalized_edit(ts1, ts2)
    feats["sorted_tokens_jaccard"] = _jaccard_tokens(ts1, ts2)

    feats["core_exact_match"] = _exact_match(c1, c2)
    feats["name_norm_exact"] = _exact_match(n1, n2)
    feats["acronym_match"] = _acronym_match(ac1, ac2)
    feats["legal_suffix_agreement"] = _legal_suffix_agreement(n1, n2)

    # ── Address features ───────────────────────────────────────────────────
    a1 = row_s1.get("addr_norm", "") or ""
    a2 = row_cand.get("addr_norm", "") or ""
    at1 = row_s1.get("addr_tokens", "") or ""
    at2 = row_cand.get("addr_tokens", "") or ""
    p1 = row_s1.get("postal_code", "") or ""
    p2 = row_cand.get("postal_code", "") or ""
    sn1 = row_s1.get("street_num", "") or ""
    sn2 = row_cand.get("street_num", "") or ""

    feats["addr_edit_dist"] = _normalized_edit(a1, a2)
    feats["addr_token_overlap"] = _addr_token_overlap(at1, at2)
    feats["addr_partial_ratio"] = _partial_ratio(a1, a2)
    feats["addr_token_set_ratio"] = _token_set_ratio(a1, a2)
    feats["addr_jaro_winkler"] = _safe_jw(a1, a2)

    feats["postal_match"] = _postal_match(p1, p2)
    feats["postal_both_missing"] = int(not p1 and not p2)
    feats["postal_one_missing"] = int(bool(p1) != bool(p2))

    feats["street_num_match"] = _street_num_match(sn1, sn2)
    feats["street_num_both_missing"] = int(not sn1 and not sn2)

    feats["has_landmark_s1"] = row_s1.get("has_landmark", 0)
    feats["has_landmark_cand"] = row_cand.get("has_landmark", 0)

    # Same country
    co1 = (row_s1.get("country", "") or "").strip().lower()
    co2 = (row_cand.get("country", "") or "").strip().lower()
    feats["same_country"] = int(co1 == co2)

    # ── Semantic embedding features ────────────────────────────────────────
    if s1_emb is not None and cand_emb is not None:
        cos_sim = float(np.dot(s1_emb, cand_emb))
        feats["emb_cosine"] = cos_sim
    else:
        feats["emb_cosine"] = np.nan

    # ── Meta features ──────────────────────────────────────────────────────
    feats["name_s1_empty"] = int(not n1)
    feats["name_cand_empty"] = int(not n2)
    feats["addr_s1_empty"] = int(not a1)
    feats["addr_cand_empty"] = int(not a2)

    # Source: S2 vs S3
    cand_id = row_cand.get("entity_id", "")
    feats["is_s2"] = int(str(cand_id).startswith("S2-"))
    feats["is_s3"] = int(str(cand_id).startswith("S3-"))

    return feats


FEATURE_NAMES = [
    "name_jaccard_tokens", "name_jaccard_chars3", "name_jaccard_chars4",
    "name_edit_dist", "name_jaro_winkler", "name_partial_ratio",
    "name_token_sort_ratio", "name_token_set_ratio",
    "core_jaccard_tokens", "core_edit_dist", "core_jaro_winkler",
    "core_partial_ratio", "core_token_sort_ratio", "core_token_set_ratio",
    "sorted_tokens_exact", "sorted_tokens_edit", "sorted_tokens_jaccard",
    "core_exact_match", "name_norm_exact", "acronym_match", "legal_suffix_agreement",
    "addr_edit_dist", "addr_token_overlap", "addr_partial_ratio",
    "addr_token_set_ratio", "addr_jaro_winkler",
    "postal_match", "postal_both_missing", "postal_one_missing",
    "street_num_match", "street_num_both_missing",
    "has_landmark_s1", "has_landmark_cand",
    "same_country",
    "emb_cosine",
    "name_s1_empty", "name_cand_empty", "addr_s1_empty", "addr_cand_empty",
    "is_s2", "is_s3",
    # Context features added dynamically after scoring:
    # "cand_rank", "cand_score_gap", "n_s1_competing"
]


def build_feature_matrix(
    candidates_df: pd.DataFrame,
    norm_dict: dict,
    emb_dict: Optional[dict] = None,
    add_context: bool = True,
) -> pd.DataFrame:
    """
    Build feature matrix for all candidate pairs.

    Parameters
    ----------
    candidates_df : DataFrame with source1_entity_id, candidate_entity_id
    norm_dict     : dict mapping entity_id → normalised record dict
    emb_dict      : dict mapping entity_id → embedding vector (optional)
    add_context   : whether to compute context features (ranking within S1 group)

    Returns
    -------
    DataFrame with all features + source1_entity_id + candidate_entity_id columns
    """
    rows = []
    logger.info(f"Building features for {len(candidates_df)} pairs ...")

    for _, pair in candidates_df.iterrows():
        s1_id = pair["source1_entity_id"]
        cand_id = pair["candidate_entity_id"]

        row_s1 = norm_dict.get(s1_id, {})
        row_cand = norm_dict.get(cand_id, {})

        s1_emb = emb_dict.get(s1_id) if emb_dict else None
        cand_emb = emb_dict.get(cand_id) if emb_dict else None

        feats = compute_pair_features(row_s1, row_cand, s1_emb, cand_emb)
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = cand_id
        rows.append(feats)

    feat_df = pd.DataFrame(rows)

    if add_context and "emb_cosine" in feat_df.columns:
        # Context: rank of candidate within its S1 group (by embedding cosine)
        feat_df["cand_rank"] = feat_df.groupby("source1_entity_id")["emb_cosine"].rank(
            ascending=False, method="dense"
        )
        # Score gap to best candidate in group
        group_max = feat_df.groupby("source1_entity_id")["emb_cosine"].transform("max")
        feat_df["cand_score_gap"] = group_max - feat_df["emb_cosine"]
        # Number of candidates per S1
        feat_df["n_s1_candidates"] = feat_df.groupby("source1_entity_id")[
            "candidate_entity_id"].transform("count")

    return feat_df


def build_feature_matrix_chunked(
    candidates_df: pd.DataFrame,
    norm_dict: dict,
    emb_dict: Optional[dict] = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Memory-efficient version that processes in chunks then concatenates."""
    chunks = []
    n = len(candidates_df)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = candidates_df.iloc[start:end]
        feat_chunk = build_feature_matrix(chunk, norm_dict, emb_dict, add_context=False)
        chunks.append(feat_chunk)
        if (start // chunk_size) % 10 == 0:
            logger.info(f"  Features: {end}/{n} pairs processed")

    feat_df = pd.concat(chunks, ignore_index=True)

    # Add context features on full data
    if "emb_cosine" in feat_df.columns:
        feat_df["cand_rank"] = feat_df.groupby("source1_entity_id")["emb_cosine"].rank(
            ascending=False, method="dense"
        )
        group_max = feat_df.groupby("source1_entity_id")["emb_cosine"].transform("max")
        feat_df["cand_score_gap"] = group_max - feat_df["emb_cosine"]
        feat_df["n_s1_candidates"] = feat_df.groupby("source1_entity_id")[
            "candidate_entity_id"].transform("count")

    return feat_df
