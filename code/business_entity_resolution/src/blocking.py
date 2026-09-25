"""
blocking.py
===========
Candidate-pair generation (blocking) stage.

Goal:
    For every S1 entity, return a shortlist of S2/S3 candidate IDs
    that might be the same business.

Blockers used (union):
    1. Country filter  – only compare within same country string
    2. Postal code     – same PIN/ZIP code
    3. Name TF-IDF     – character 3-gram TF-IDF nearest neighbours
    4. Embedding ANN   – multilingual sentence-transformer FAISS top-K
    5. Rare-token      – share a rare (high-IDF) name token
    6. City token      – share the same city-like address tokens

Memory-safety changes:
    - Blocker outputs are streamed instead of building large temporary
      Dict[str, Set[str]] objects.
    - Rare-token postings store integer row indices rather than entity
      ID strings.
    - TF-IDF similarity remains sparse; no .toarray().
    - Temporary objects are explicitly released between blockers.
    - Original blocker thresholds are preserved.
"""

import gc
import heapq
import itertools
import logging
import os
import pickle
import shutil
import sqlite3
from array import array
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Iterator

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TFIDF_TOPK = 30
EMBED_TOPK = 25

# Original rare-token definition.
# IMPORTANT: do not add an artificial posting cap here.
RARE_TOKEN_MAX_DF = 0.01
MIN_TOKEN_LEN = 3

EMBED_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

EMBED_BATCH = 512
EMBED_DIM = 384


# ─────────────────────────────────────────────────────────────────────────────
# Cache directory
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    ".cache",
)

os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(name: str) -> str:
    return os.path.join(_CACHE_DIR, name)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_combined_text(df: pd.DataFrame) -> pd.Series:
    """Build 'name_norm + addr_norm' text for embedding."""
    return (
        df["name_norm"].fillna("")
        + " "
        + df["addr_norm"].fillna("")
    ).str.strip()


def _get_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    cache_key: Optional[str] = None,
) -> np.ndarray:
    """
    Compute or load cached sentence embeddings.
    """

    cache_file: Optional[str] = None

    if cache_key:
        cache_file = _cache_path(
            f"{cache_key}_embeddings.npy"
        )

        if os.path.exists(cache_file):
            logger.info(
                f"Loading cached embeddings: {cache_key}"
            )
            return np.load(cache_file)

    logger.info(
        f"Encoding {len(texts)} texts ..."
    )

    embs = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    if cache_file is not None:
        np.save(cache_file, embs)

    return embs


# ─────────────────────────────────────────────────────────────────────────────
# Blocker 1: Postal code
# ─────────────────────────────────────────────────────────────────────────────

def _iter_postal_pairs(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> Iterator[Tuple[int, int]]:
    """
    Yield all S1/candidate pairs sharing a non-empty postal code.

    This replaces the old temporary:
        Dict[S1_ID, Set[candidate_ID]]

    with a streaming generator.
    """

    postal_idx: Dict[str, array] = defaultdict(lambda: array("i"))

    # Candidate postal-code index.
    for cand_idx, row in enumerate(cand_df.itertuples(index=False)):
        postal = row.postal_code

        if postal:
            postal_idx[postal].append(cand_idx)

    # Emit pairs directly.
    for s1_idx, row in enumerate(s1_df.itertuples(index=False)):
        postal = row.postal_code

        if not postal:
            continue

        matching_ids = postal_idx.get(postal)

        if matching_ids:
            for cand_idx in matching_ids:
                yield s1_idx, cand_idx

    del postal_idx


# ─────────────────────────────────────────────────────────────────────────────
# Blocker 2: Rare token
# ─────────────────────────────────────────────────────────────────────────────

def _iter_rare_token_pairs(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    max_df: float = RARE_TOKEN_MAX_DF,
) -> Iterator[Tuple[int, int]]:
    """
    Yield candidate pairs sharing a rare, informative name token.

    IMPORTANT:
        The original rare-token rule is preserved:

            token DF <= 1%
            token must occur at least twice

        There is NO artificial posting-list cap.

    Memory optimization:
        candidate postings store integer row indices rather than
        repeatedly storing candidate entity-id strings.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Pass 1: calculate token document frequency
    # ─────────────────────────────────────────────────────────────────────

    token_freq: Counter = Counter()

    for name in cand_df["name_core"].fillna(""):
        for tok in set(name.split()):
            if len(tok) >= MIN_TOKEN_LEN:
                token_freq[tok] += 1

    for name in s1_df["name_core"].fillna(""):
        for tok in set(name.split()):
            if len(tok) >= MIN_TOKEN_LEN:
                token_freq[tok] += 1

    total = len(cand_df) + len(s1_df)

    rare_tokens = {
        tok
        for tok, cnt in token_freq.items()
        if (
            cnt >= 2
            and (cnt / max(total, 1)) <= max_df
        )
    }

    logger.info(
        f"Rare-token vocabulary: {len(rare_tokens)} tokens"
    )

    # Token frequency dictionary is no longer needed.
    del token_freq
    gc.collect()

    # ─────────────────────────────────────────────────────────────────────
    # Pass 2: token -> candidate row indices
    # ─────────────────────────────────────────────────────────────────────

    tok_to_cands: Dict[str, array] = defaultdict(lambda: array("i"))

    for idx, name in enumerate(
        cand_df["name_core"].fillna("")
    ):
        for tok in set(name.split()):
            if tok in rare_tokens:
                tok_to_cands[tok].append(idx)

    # ─────────────────────────────────────────────────────────────────────
    # Pass 3: stream S1 -> candidate pairs
    # ─────────────────────────────────────────────────────────────────────

    for s1_idx, row in enumerate(s1_df.itertuples(index=False)):

        name = row.name_core or ""

        for tok in set(name.split()):

            posting = tok_to_cands.get(tok)

            if not posting:
                continue

            for cand_idx in posting:
                yield s1_idx, cand_idx

    del tok_to_cands
    del rare_tokens
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Blocker 3: TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def _iter_tfidf_pairs(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    topk: int = TFIDF_TOPK,
) -> Iterator[Tuple[int, int]]:
    """
    Character 3-gram TF-IDF sparse cosine retrieval.

    Memory-safe version:
        - uses sparse similarity matrices
        - never calls .toarray()
        - processes S1 in small chunks
        - only keeps the strongest top-K non-zero similarities
    """

    s1_names = list(
        s1_df["name_core"].fillna("")
    )

    cand_names = list(
        cand_df["name_core"].fillna("")
    )

    logger.info("Fitting TF-IDF ...")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    all_names = s1_names + cand_names

    vectorizer.fit(all_names)

    # Free the combined Python list.
    del all_names
    gc.collect()

    s1_mat = vectorizer.transform(
        s1_names
    )

    cand_mat = vectorizer.transform(
        cand_names
    )

    # Normalize for cosine similarity.
    s1_mat = sk_normalize(
        s1_mat,
        norm="l2",
    )

    cand_mat = sk_normalize(
        cand_mat,
        norm="l2",
    )

    logger.info(
        f"TF-IDF matrices: "
        f"S1={s1_mat.shape}, "
        f"Cand={cand_mat.shape}"
    )

    # Smaller chunk avoids large temporary sparse products.
    CHUNK = 128

    for i in tqdm(
        range(0, len(s1_df), CHUNK),
        desc="TF-IDF blocking",
    ):

        chunk_s1 = s1_mat[
            i:i + CHUNK
        ]

        # IMPORTANT:
        # Keep the result sparse.
        sims = chunk_s1 @ cand_mat.T
        sims = sims.tocsr()

        for j in range(
            sims.shape[0]
        ):

            row = sims.getrow(j)

            if row.nnz == 0:
                continue

            data = row.data
            indices = row.indices

            k = min(
                topk,
                len(data),
            )

            if len(data) > k:

                top_positions = np.argpartition(
                    data,
                    -k,
                )[-k:]

            else:

                top_positions = np.arange(
                    len(data)
                )

            # Sort strongest first.
            top_positions = top_positions[
                np.argsort(
                    data[top_positions]
                )[::-1]
            ]

            for pos in top_positions:

                sim = float(
                    data[pos]
                )

                # Preserve original threshold.
                if sim <= 0.05:
                    continue

                col = indices[pos]

                yield (
                    i + j,
                    int(col),
                )

        del sims
        del chunk_s1

    del s1_mat
    del cand_mat
    del s1_names
    del cand_names
    del vectorizer

    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Blocker 4: Embedding ANN
# ─────────────────────────────────────────────────────────────────────────────

def _iter_embedding_pairs(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    model: SentenceTransformer,
    topk: int = EMBED_TOPK,
    s1_cache: Optional[str] = None,
    cand_cache: Optional[str] = None,
) -> Iterator[Tuple[int, int]]:
    """
    Multilingual sentence embedding ANN retrieval using FAISS.
    """

    s1_texts = list(
        _build_combined_text(s1_df)
    )

    cand_texts = list(
        _build_combined_text(cand_df)
    )

    # Compute/load embeddings.
    s1_embs = _get_embeddings(
        s1_texts,
        model,
        s1_cache,
    )

    cand_embs = _get_embeddings(
        cand_texts,
        model,
        cand_cache,
    )

    logger.info(
        f"Building FAISS index "
        f"({len(cand_df)} vectors, "
        f"dim={cand_embs.shape[1]}) ..."
    )

    # Candidate vectors are normalized, so inner product == cosine similarity.
    index = faiss.IndexFlatIP(
        cand_embs.shape[1]
    )

    index.add(
        cand_embs.astype(
            np.float32,
            copy=False,
        )
    )

    logger.info(
        f"Searching top-{topk} neighbours ..."
    )

    D, I = index.search(
        s1_embs.astype(
            np.float32,
            copy=False,
        ),
        topk,
    )

    # Stream results immediately.
    for i, (
        dists,
        idxs,
    ) in enumerate(
        zip(D, I)
    ):

        for dist, idx in zip(
            dists,
            idxs,
        ):

            # Preserve original threshold.
            if (
                idx >= 0
                and dist > 0.3
            ):
                yield (
                    i,
                    int(idx),
                )

    # Release very large objects.
    del D
    del I
    del index
    del s1_embs
    del cand_embs
    del s1_texts
    del cand_texts

    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Blocker 5: City token
# ─────────────────────────────────────────────────────────────────────────────

def _extract_city(addr_norm: str) -> str:
    """
    Extract the last 2 significant non-numeric address tokens.
    """

    if not addr_norm:
        return ""

    parts = [
        p
        for p in addr_norm.split()
        if not p.isdigit()
        and len(p) >= 3
    ]

    if len(parts) >= 2:
        return " ".join(
            parts[-2:]
        )

    if parts:
        return " ".join(parts)

    return ""


def _iter_city_pairs(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> Iterator[Tuple[int, int]]:
    """
    Yield pairs sharing the same city-like address tokens.
    """

    city_idx: Dict[str, array] = defaultdict(lambda: array("i"))

    for cand_idx, row in enumerate(cand_df.itertuples(index=False)):

        city = _extract_city(
            row.addr_norm
        )

        if city:
            city_idx[city].append(
                cand_idx
            )

    for s1_idx, row in enumerate(s1_df.itertuples(index=False)):

        city = _extract_city(
            row.addr_norm
        )

        if not city:
            continue

        matching_ids = city_idx.get(city)

        if matching_ids:

            for cand_idx in matching_ids:
                yield s1_idx, cand_idx

    del city_idx


# ─────────────────────────────────────────────────────────────────────────────
# Main blocking function
# ─────────────────────────────────────────────────────────────────────────────

def _legacy_generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    use_embeddings: bool = True,
    split_name: str = "train",
) -> pd.DataFrame:
    """
    Generate candidate pairs by unioning multiple blockers.

    Parameters
    ----------
    s1_df, s2_df, s3_df:
        Normalised DataFrames.

    use_embeddings:
        Whether to use sentence-transformer ANN blocker.

    split_name:
        'train' or 'test', used for embedding cache keys.

    Returns
    -------
    DataFrame
        Columns:
            source1_entity_id
            candidate_entity_id
    """

    model = None

    # ─────────────────────────────────────────────────────────────────────
    # Load embedding model once.
    # ─────────────────────────────────────────────────────────────────────

    if use_embeddings:

        logger.info(
            f"Loading embedding model: "
            f"{EMBED_MODEL}"
        )

        model = SentenceTransformer(
            EMBED_MODEL
        )

    # Combine S2 and S3 candidates.
    cand_df = pd.concat(
        [
            s2_df,
            s3_df,
        ],
        ignore_index=True,
    )

    # Master union.
    #
    # This is the one structure that remains alive across blockers.
    # Individual blockers no longer create separate huge dictionaries.
    all_candidates: Dict[
        str,
        Set[str],
    ] = defaultdict(set)

    # ─────────────────────────────────────────────────────────────────────
    # Per-country blocking
    # ─────────────────────────────────────────────────────────────────────

    countries = s1_df[
        "country"
    ].unique()

    logger.info(
        f"Countries in S1: "
        f"{list(countries)}"
    )

    for country in countries:

        logger.info(
            f"\n=== Blocking for country: "
            f"{country} ==="
        )

        mask_s1 = (
            s1_df["country"]
            == country
        )

        mask_cand = (
            cand_df["country"]
            == country
        )

        s1_c = s1_df[
            mask_s1
        ].reset_index(
            drop=True
        )

        cand_c = cand_df[
            mask_cand
        ].reset_index(
            drop=True
        )

        if (
            len(s1_c) == 0
            or len(cand_c) == 0
        ):

            logger.info(
                f"  Skipping {country}: "
                f"empty partition"
            )

            del s1_c
            del cand_c
            gc.collect()

            continue

        logger.info(
            f"  S1: {len(s1_c)}, "
            f"Cand: {len(cand_c)}"
        )

        # ================================================================
        # Blocker 1: Postal code
        # ================================================================

        logger.info(
            "  Running postal-code blocker ..."
        )

        postal_pair_count = 0

        for (
            s1_id,
            cand_id,
        ) in _iter_postal_pairs(
            s1_c,
            cand_c,
        ):

            all_candidates[
                s1_id
            ].add(cand_id)

            postal_pair_count += 1

        logger.info(
            f"  Postal: "
            f"{postal_pair_count} pairs"
        )

        gc.collect()

        # ================================================================
        # Blocker 2: Rare token
        # ================================================================

        logger.info(
            "  Running rare-token blocker ..."
        )

        rare_pair_count = 0

        for (
            s1_id,
            cand_id,
        ) in _iter_rare_token_pairs(
            s1_c,
            cand_c,
        ):

            all_candidates[
                s1_id
            ].add(cand_id)

            rare_pair_count += 1

        logger.info(
            f"  Rare-token: "
            f"{rare_pair_count} pairs"
        )

        gc.collect()

        # ================================================================
        # Blocker 3: TF-IDF
        # ================================================================

        logger.info(
            "  Running TF-IDF blocker ..."
        )

        tfidf_pair_count = 0

        for (
            s1_id,
            cand_id,
        ) in _iter_tfidf_pairs(
            s1_c,
            cand_c,
        ):

            all_candidates[
                s1_id
            ].add(cand_id)

            tfidf_pair_count += 1

        logger.info(
            f"  TF-IDF: "
            f"{tfidf_pair_count} pairs"
        )

        gc.collect()

        # ================================================================
        # Blocker 4: Embedding ANN
        # ================================================================

        if (
            use_embeddings
            and model is not None
        ):

            logger.info(
                "  Running embedding ANN blocker ..."
            )

            embedding_pair_count = 0

            for (
                s1_id,
                cand_id,
            ) in _iter_embedding_pairs(
                s1_c,
                cand_c,
                model,
                topk=EMBED_TOPK,
                s1_cache=(
                    f"{split_name}_{country}_s1"
                ),
                cand_cache=(
                    f"{split_name}_{country}_cand"
                ),
            ):

                all_candidates[
                    s1_id
                ].add(cand_id)

                embedding_pair_count += 1

            logger.info(
                f"  Embedding: "
                f"{embedding_pair_count} pairs"
            )

            gc.collect()

        # ================================================================
        # Blocker 5: City token
        # ================================================================

        logger.info(
            "  Running city-token blocker ..."
        )

        city_pair_count = 0

        for (
            s1_id,
            cand_id,
        ) in _iter_city_pairs(
            s1_c,
            cand_c,
        ):

            all_candidates[
                s1_id
            ].add(cand_id)

            city_pair_count += 1

        logger.info(
            f"  City: "
            f"{city_pair_count} pairs"
        )

        # Release country-specific DataFrames.
        del s1_c
        del cand_c
        del mask_s1
        del mask_cand

        gc.collect()

    # ─────────────────────────────────────────────────────────────────────
    # Unknown / empty-country records
    # ─────────────────────────────────────────────────────────────────────

    mask_empty_s1 = (
        s1_df["country"].isin(
            ["", "unknown"]
        )
        | s1_df["country"].isna()
    )

    if mask_empty_s1.any():

        logger.info(
            "Running blocker for "
            "unknown-country S1 records ..."
        )

        s1_unk = s1_df[
            mask_empty_s1
        ].reset_index(
            drop=True
        )

        unknown_pair_count = 0

        for (
            s1_id,
            cand_id,
        ) in _iter_tfidf_pairs(
            s1_unk,
            cand_df,
        ):

            all_candidates[
                s1_id
            ].add(cand_id)

            unknown_pair_count += 1

        logger.info(
            f"  Unknown-country TF-IDF: "
            f"{unknown_pair_count} pairs"
        )

        del s1_unk
        gc.collect()

    # ─────────────────────────────────────────────────────────────────────
    # Ensure every S1 entity exists.
    # ─────────────────────────────────────────────────────────────────────

    for s1_id in s1_df[
        "entity_id"
    ]:

        _ = all_candidates[
            s1_id
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Flatten master dictionary.
    #
    # We POP items while flattening so we don't keep both:
    #
    #     all_candidates
    #          +
    #     huge rows list
    #
    # alive at the same time.
    # ─────────────────────────────────────────────────────────────────────

    rows: List[
        Tuple[str, str]
    ] = []

    total_candidate_pairs = 0

    while all_candidates:

        s1_id, cand_set = (
            all_candidates.popitem()
        )

        total_candidate_pairs += len(
            cand_set
        )

        for cand_id in cand_set:

            rows.append(
                (
                    s1_id,
                    cand_id,
                )
            )

    logger.info(
        f"\nTotal candidate pairs: "
        f"{total_candidate_pairs} "
        f"(avg "
        f"{total_candidate_pairs / max(len(s1_df), 1):.1f} "
        f"per S1)"
    )

    # Master dictionary can now be released completely.
    del all_candidates
    gc.collect()

    # ─────────────────────────────────────────────────────────────────────
    # Build final DataFrame.
    # ─────────────────────────────────────────────────────────────────────

    result_df = pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
    )

    del rows
    del cand_df
    del model

    gc.collect()

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Memory-safe country-partitioned blocking
# ─────────────────────────────────────────────────────────────────────────────

def _write_candidate_chunk(
    pairs: np.ndarray,
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    path: str,
) -> None:
    """Convert one compact pair array to IDs and atomically write a chunk."""
    if pairs.size == 0:
        return

    s1_ids = s1_df["entity_id"].to_numpy(dtype=object)
    cand_ids = cand_df["entity_id"].to_numpy(dtype=object)
    output = pd.DataFrame({
        "source1_entity_id": s1_ids[pairs["s1"]],
        "candidate_entity_id": cand_ids[pairs["cand"]],
    })
    temp_path = f"{path}.tmp"
    output.to_parquet(temp_path, index=False)
    os.replace(temp_path, path)
    del output, s1_ids, cand_ids


def _iter_compact_pair_chunks(
    iterator: Iterator[Tuple[int, int]],
    candidate_count: int,
) -> Iterator[np.ndarray]:
    """Yield bounded, deduplicated structured arrays from one blocker."""
    pair_dtype = np.dtype([("s1", np.int32), ("cand", np.int32)])
    while True:
        encoded = np.fromiter(
            (s1_idx * candidate_count + cand_idx
             for s1_idx, cand_idx in itertools.islice(iterator, 250_000)),
            dtype=np.int64,
        )
        if not len(encoded):
            return
        pairs = np.empty(len(encoded), dtype=pair_dtype)
        pairs["s1"] = encoded // candidate_count
        pairs["cand"] = encoded % candidate_count
        del encoded
        yield np.unique(pairs)
        del pairs


def _iter_merged_pair_chunks(
    run_paths: List[str],
    candidate_count: int,
) -> Iterator[np.ndarray]:
    """Merge sorted pair runs on disk, removing duplicates with bounded RAM."""
    if not run_paths:
        return
    runs = [np.load(path, mmap_mode="r") for path in run_paths]
    heap = []
    for run_index, run in enumerate(runs):
        if len(run):
            pair = run[0]
            heapq.heappush(
                heap,
                (int(pair["s1"]) * candidate_count + int(pair["cand"]), run_index, 0),
            )

    output = np.empty(100_000, dtype=[("s1", np.int32), ("cand", np.int32)])
    output_count = 0
    last_key = None
    while heap:
        key, run_index, position = heapq.heappop(heap)
        if key != last_key:
            output["s1"][output_count] = key // candidate_count
            output["cand"][output_count] = key % candidate_count
            output_count += 1
            last_key = key
            if output_count == len(output):
                yield output.copy()
                output_count = 0
        next_position = position + 1
        if next_position < len(runs[run_index]):
            pair = runs[run_index][next_position]
            heapq.heappush(
                heap,
                (int(pair["s1"]) * candidate_count + int(pair["cand"]),
                 run_index, next_position),
            )
    if output_count:
        yield output[:output_count].copy()
    del runs, heap, output


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    use_embeddings: bool = True,
    split_name: str = "train",
) -> pd.DataFrame:
    """Generate candidates with compact per-country integer pair storage."""
    model = SentenceTransformer(EMBED_MODEL, device="cuda:0") if use_embeddings else None
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    chunk_dir = os.path.join(_CACHE_DIR, "candidate_chunks", split_name)
    os.makedirs(chunk_dir, exist_ok=True)
    for old_path in os.listdir(chunk_dir):
        full_old_path = os.path.join(chunk_dir, old_path)
        if old_path.endswith((".parquet", ".tmp", ".sqlite")):
            os.remove(full_old_path)
        elif old_path.endswith("_runs") and os.path.isdir(full_old_path):
            shutil.rmtree(full_old_path)

    chunk_paths: List[str] = []
    country_values = list(s1_df["country"].drop_duplicates())

    def process_partition(country: str, s1_c: pd.DataFrame, cand_c: pd.DataFrame, chunk_name: str,
                          country_blockers: bool = True) -> None:
        if s1_c.empty or cand_c.empty:
            return
        run_paths: List[str] = []

        def add_blocker(name: str, iterator: Iterator[Tuple[int, int]]) -> None:
            blocker_count = 0
            run_index = 0
            for pairs in _iter_compact_pair_chunks(iterator, len(cand_c)):
                run_path = os.path.join(
                    run_dir, f"{chunk_name}_{name.lower()}_{run_index}.npy"
                )
                np.save(run_path, pairs, allow_pickle=False)
                run_paths.append(run_path)
                run_index += 1
                blocker_count += len(pairs)
                del pairs
            logger.info("  %s: %d emitted pairs", name, blocker_count)
            gc.collect()

        run_dir = os.path.join(chunk_dir, f"{chunk_name}_runs")
        os.makedirs(run_dir, exist_ok=True)

        if country_blockers:
            add_blocker("Postal", _iter_postal_pairs(s1_c, cand_c))
            add_blocker("Rare-token", _iter_rare_token_pairs(s1_c, cand_c))
        add_blocker("TF-IDF", _iter_tfidf_pairs(s1_c, cand_c))
        if model is not None and country_blockers:
            add_blocker(
                "Embedding",
                _iter_embedding_pairs(
                    s1_c, cand_c, model, topk=EMBED_TOPK,
                    s1_cache=f"{split_name}_{chunk_name}_s1",
                    cand_cache=f"{split_name}_{chunk_name}_cand",
                ),
            )
        if country_blockers:
            add_blocker("City", _iter_city_pairs(s1_c, cand_c))

        if run_paths:
            chunk_path = os.path.join(chunk_dir, f"{chunk_name}.parquet")
            temp_path = f"{chunk_path}.tmp"
            import pyarrow as pa
            import pyarrow.parquet as pq
            s1_ids = s1_c["entity_id"].to_numpy(dtype=object)
            cand_ids = cand_c["entity_id"].to_numpy(dtype=object)
            writer = None
            for merged_pairs in _iter_merged_pair_chunks(run_paths, len(cand_c)):
                output = pd.DataFrame({
                    "source1_entity_id": s1_ids[merged_pairs["s1"]],
                    "candidate_entity_id": cand_ids[merged_pairs["cand"]],
                })
                table = pa.Table.from_pandas(output, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp_path, table.schema)
                writer.write_table(table)
                del merged_pairs, output, table
            if writer is not None:
                writer.close()
            os.replace(temp_path, chunk_path)
            chunk_paths.append(chunk_path)
            del s1_ids, cand_ids
        for run_path in run_paths:
            os.remove(run_path)
        os.rmdir(run_dir)
        gc.collect()

    for country_index, country in enumerate(country_values):
        s1_c = s1_df[s1_df["country"] == country].reset_index(drop=True)
        cand_c = cand_df[cand_df["country"] == country].reset_index(drop=True)
        process_partition(str(country), s1_c, cand_c, f"country_{country_index}")
        del s1_c, cand_c
        gc.collect()

    empty_mask = s1_df["country"].isin(["", "unknown"]) | s1_df["country"].isna()
    if empty_mask.any():
        s1_unknown = s1_df[empty_mask].reset_index(drop=True)
        process_partition("unknown", s1_unknown, cand_df, "unknown", country_blockers=False)
        del s1_unknown
        gc.collect()

    if not chunk_paths:
        result = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    else:
        result = pd.concat(
            [pd.read_parquet(path) for path in chunk_paths],
            ignore_index=True,
        )
        result = result.drop_duplicates(ignore_index=True)

    logger.info("Total candidate pairs: %d (%.1f per S1)", len(result), len(result) / max(len(s1_df), 1))
    del cand_df, model, chunk_paths
    gc.collect()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Blocking evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_blocking(
    candidates_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> dict:
    """
    Measure blocking pair completeness / recall
    and reduction ratio.
    """

    gt_pairs: Set[
        Tuple[str, str]
    ] = set()

    for row in ground_truth_df.itertuples(
        index=False
    ):

        s1_id = row.source1_entity_id

        matched = str(
            getattr(
                row,
                "matched_entity_ids",
                "",
            )
            or ""
        )

        for mid in matched.split(","):

            mid = mid.strip()

            if mid:
                gt_pairs.add(
                    (
                        s1_id,
                        mid,
                    )
                )

    cand_pairs: Set[
        Tuple[str, str]
    ] = set(
        zip(
            candidates_df[
                "source1_entity_id"
            ],
            candidates_df[
                "candidate_entity_id"
            ],
        )
    )

    tp = (
        gt_pairs
        & cand_pairs
    )

    recall = (
        len(tp)
        / max(
            len(gt_pairs),
            1,
        )
    )

    total_possible = (
        len(
            candidates_df[
                "source1_entity_id"
            ].unique()
        )
        *
        len(
            candidates_df[
                "candidate_entity_id"
            ].unique()
        )
    )

    reduction_ratio = (
        1
        -
        len(cand_pairs)
        /
        max(
            total_possible,
            1,
        )
    )

    return {
        "pair_completeness_recall": recall,
        "true_pairs": len(gt_pairs),
        "found_pairs": len(tp),
        "missed_pairs": (
            len(gt_pairs)
            - len(tp)
        ),
        "candidate_pairs": len(
            cand_pairs
        ),
        "reduction_ratio": reduction_ratio,
    }