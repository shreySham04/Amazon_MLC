"""
predict.py
==========
Inference pipeline: load trained model and generate output TSVs for the test set.

Steps:
  1. Load & normalise test sources
  2. Generate candidate pairs (same blockers as training)
  3. Build feature matrix
  4. Score with LightGBM
  5. Apply decision threshold + post-processing
     - Per-entity thresholding (output empty if no candidate clears)
     - Conflict resolution: if one S2/S3 ID claimed by multiple S1s, assign to best-scoring one
  6. Write matching_results.tsv and candidate_pairs.tsv
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from blocking import generate_candidates
from features import build_feature_matrix_chunked
from normalize import normalize_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[3].resolve()   # student_resource/
DATA_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = (Path(__file__).parent / ".." / "models").resolve()
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_and_normalize(path: Path, source_prefix: str) -> Tuple[pd.DataFrame, Dict]:
    """Load TSV, normalise records, return DataFrame + entity_id → record dict."""
    cache_path = MODEL_DIR / f"normalized_test_{source_prefix.lower()}.parquet"
    required_columns = {
        "entity_id", "name_norm", "name_core", "name_tokens_sorted",
        "name_acronym", "addr_norm", "addr_tokens", "postal_code",
        "street_num", "has_landmark", "country",
    }
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if required_columns.issubset(cached.columns) and len(cached) > 0:
                cached = cached.fillna("")
                records = cached.to_dict(orient="records")
                return cached, {record["entity_id"]: record for record in records}
            logger.warning("Invalid normalization cache %s; rebuilding", cache_path)
        except Exception as exc:
            logger.warning("Unreadable normalization cache %s: %s; rebuilding", cache_path, exc)
        cache_path.unlink(missing_ok=True)

    logger.info(f"Loading {path.name} ...")
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")

    norm_records = []
    norm_dict = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Normalising {source_prefix}"):
        rec = normalize_record(
            row["entity_id"], row["business_name"],
            row["business_address"], row["country"]
        )
        norm_records.append(rec)
        norm_dict[row["entity_id"]] = rec

    norm_df = pd.DataFrame(norm_records)
    temp_path = cache_path.with_name(cache_path.stem + ".tmp.parquet")
    norm_df.to_parquet(temp_path, index=False)
    os.replace(temp_path, cache_path)
    logger.info("Saved normalized cache to %s", cache_path)
    return norm_df, norm_dict


def compute_test_embeddings(norm_dict: Dict, entity_ids: List[str],
                            embed_model) -> Dict[str, np.ndarray]:
    """Compute embeddings for test entity IDs (no cache reuse across train/test)."""
    cache_path = MODEL_DIR / "test_all_emb_dict.pkl"
    if cache_path.exists():
        logger.info("Loading cached test embeddings ...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from blocking import EMBED_BATCH
    texts = []
    valid_ids = []
    for eid in entity_ids:
        rec = norm_dict.get(eid, {})
        text = (rec.get("name_norm", "") + " " + rec.get("addr_norm", "")).strip()
        texts.append(text)
        valid_ids.append(eid)

    logger.info(f"Computing test embeddings for {len(texts)} records ...")
    embs = embed_model.encode(
        texts, batch_size=EMBED_BATCH, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    )
    emb_dict = dict(zip(valid_ids, embs))

    with open(cache_path, "wb") as f:
        pickle.dump(emb_dict, f)

    return emb_dict


def resolve_conflicts(
    s1_to_matches: Dict[str, Set[str]],
    s1_to_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Set[str]]:
    """
    If the same S2/S3 ID is claimed by multiple S1 entities,
    assign it only to the one with the highest match score.
    """
    # Build: cand_id → list of (s1_id, score)
    cand_claims: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id, matches in s1_to_matches.items():
        for cand_id in matches:
            score = s1_to_scores.get(s1_id, {}).get(cand_id, 0.0)
            if cand_id not in cand_claims:
                cand_claims[cand_id] = []
            cand_claims[cand_id].append((s1_id, score))

    # For contested candidates, keep only best-scoring S1
    contested = {cid: claims for cid, claims in cand_claims.items() if len(claims) > 1}
    if contested:
        logger.info(f"Resolving {len(contested)} contested candidates ...")

    reassigned: Dict[str, Set[str]] = {s1_id: set(matches) for s1_id, matches in s1_to_matches.items()}

    for cand_id, claims in contested.items():
        best_s1, best_score = max(claims, key=lambda x: x[1])
        for s1_id, score in claims:
            if s1_id != best_s1:
                reassigned[s1_id].discard(cand_id)

    return reassigned


# ─── Main prediction routine ──────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Amazon MLC 2026 – Prediction Pipeline")
    logger.info("=" * 60)

    # ── Load model & meta ──────────────────────────────────────────────────
    model_path = MODEL_DIR / "lgbm_model.pkl"
    meta_path = MODEL_DIR / "model_meta.json"

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run train.py first.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    threshold = meta["threshold"]
    feature_cols = meta["feature_cols"]
    logger.info(f"Loaded model | threshold={threshold:.3f} | features={len(feature_cols)}")

    # ── Step 1: Load & normalise test data ────────────────────────────────
    s1_norm_df, s1_norm_dict = load_and_normalize(DATA_DIR / "test_source1.tsv", "S1")
    s2_norm_df, s2_norm_dict = load_and_normalize(DATA_DIR / "test_source2.tsv", "S2")
    s3_norm_df, s3_norm_dict = load_and_normalize(DATA_DIR / "test_source3.tsv", "S3")

    all_norm_dict = {**s1_norm_dict, **s2_norm_dict, **s3_norm_dict}
    all_s1_ids = list(s1_norm_df["entity_id"])

    # ── Step 2: Blocking ───────────────────────────────────────────────────
    cand_cache_path = MODEL_DIR / "test_candidates.parquet"
    if cand_cache_path.exists():
        logger.info("Loading cached test candidates ...")
        candidates_df = pd.read_parquet(cand_cache_path)
    else:
        logger.info("Generating test candidates ...")
        candidates_df = generate_candidates(
            s1_norm_df, s2_norm_df, s3_norm_df,
            use_embeddings=True, split_name="test"
        )
        candidates_df.to_parquet(cand_cache_path, index=False)
        logger.info(f"Saved test candidates ({len(candidates_df)} pairs)")

    logger.info(f"Test candidates: {len(candidates_df)} pairs")

    # ── Step 3: Embeddings ─────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    from blocking import EMBED_MODEL
    embed_model = SentenceTransformer(EMBED_MODEL, device="cuda:0")

    all_ids_in_cands = (
        list(candidates_df["source1_entity_id"].unique())
        + list(candidates_df["candidate_entity_id"].unique())
    )
    emb_dict = compute_test_embeddings(all_norm_dict, all_ids_in_cands, embed_model)

    # ── Step 4: Feature matrix ─────────────────────────────────────────────
    feat_cache_path = MODEL_DIR / "test_features.parquet"
    if feat_cache_path.exists():
        logger.info("Loading cached test features ...")
        feat_df = pd.read_parquet(feat_cache_path)
    else:
        logger.info("Building test feature matrix ...")
        feat_df = build_feature_matrix_chunked(candidates_df, all_norm_dict, emb_dict)
        feat_df.to_parquet(feat_cache_path, index=False)
        logger.info(f"Saved test features ({feat_df.shape})")

    # Align feature columns with training
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = np.nan

    X_test = feat_df[feature_cols].values.astype(np.float32)

    # ── Step 5: Score ──────────────────────────────────────────────────────
    logger.info("Scoring candidate pairs ...")
    probs = model.predict_proba(X_test)[:, 1]
    feat_df["match_prob"] = probs

    # ── Step 6: Decision + post-processing ────────────────────────────────
    # Initialise dicts for every S1 entity (including those with no candidates)
    s1_to_matches: Dict[str, Set[str]] = {sid: set() for sid in all_s1_ids}
    s1_to_scores: Dict[str, Dict[str, float]] = {sid: {} for sid in all_s1_ids}
    s1_to_candidates: Dict[str, Set[str]] = {sid: set() for sid in all_s1_ids}

    # Vectorised decision: avoid slow iterrows over millions of rows
    for s1_id, group in feat_df.groupby("source1_entity_id"):
        cands = group["candidate_entity_id"].tolist()
        probs_g = group["match_prob"].tolist()
        for cand_id, prob in zip(cands, probs_g):
            s1_to_candidates[s1_id].add(cand_id)
            s1_to_scores[s1_id][cand_id] = float(prob)
            if prob >= threshold:
                s1_to_matches[s1_id].add(cand_id)

    # Conflict resolution
    s1_to_matches = resolve_conflicts(s1_to_matches, s1_to_scores)

    logger.info(
        f"\nPost-processing summary:"
        f"\n  S1 entities: {len(all_s1_ids)}"
        f"\n  S1 with matches: {sum(1 for m in s1_to_matches.values() if m)}"
        f"\n  S1 singletons: {sum(1 for m in s1_to_matches.values() if not m)}"
        f"\n  Total matched pairs: {sum(len(m) for m in s1_to_matches.values())}"
    )

    # ── Step 7: Write outputs ──────────────────────────────────────────────
    # matching_results.tsv
    match_rows = []
    for s1_id in all_s1_ids:
        matches = sorted(s1_to_matches.get(s1_id, set()))
        match_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matches),
        })

    match_df = pd.DataFrame(match_rows, columns=["source1_entity_id", "matched_entity_ids"])
    match_path = OUTPUT_DIR / "matching_results.tsv"
    match_df.to_csv(match_path, sep="\t", index=False)
    logger.info(f"\n✓ Wrote {match_path}")

    # candidate_pairs.tsv
    cand_rows = []
    for s1_id in all_s1_ids:
        candidates = sorted(s1_to_candidates.get(s1_id, set()))
        cand_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(candidates),
        })

    cand_out_df = pd.DataFrame(cand_rows, columns=["source1_entity_id", "candidate_entity_ids"])
    cand_path = OUTPUT_DIR / "candidate_pairs.tsv"
    cand_out_df.to_csv(cand_path, sep="\t", index=False)
    logger.info(f"✓ Wrote {cand_path}")

    logger.info("\nRun the validator next:")
    logger.info(
        "  python utils/validate_submission.py "
        "--matching output/matching_results.tsv "
        "--candidate output/candidate_pairs.tsv "
        "--test-dir dataset/test"
    )


if __name__ == "__main__":
    main()
