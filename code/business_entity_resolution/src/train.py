"""
train.py
========
Train the LightGBM matching classifier.

Steps:
  1. Load and normalise train sources
  2. Generate candidate pairs (blocking)
  3. Build feature matrix
  4. Create labels (positive = in ground truth, negative = in candidates but not GT)
  5. Hard-negative mining: oversample hard negatives from blocking
  6. Group-split cross-validation by S1 entity
  7. Train LightGBM with calibrated probabilities
  8. Tune decision threshold on F0.5 metric
  9. Save model, threshold, and feature list
"""

import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from blocking import generate_candidates, evaluate_blocking
from features import build_feature_matrix_chunked, FEATURE_NAMES
from normalize import normalize_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[3].resolve()   # student_resource/
DATA_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = (Path(__file__).parent / ".." / "models").resolve()
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_and_normalize(path: Path, source_prefix: str) -> Tuple[pd.DataFrame, Dict]:
    """Load TSV, normalise records, return DataFrame + entity_id → record dict."""
    logger.info(f"Loading {path.name} ...")
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    cache_path = MODEL_DIR / f"normalized_train_{source_prefix.lower()}.parquet"
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


def compute_embeddings(norm_dict: Dict, entity_ids: List[str],
                       model, cache_key: str) -> Dict[str, np.ndarray]:
    """Compute embeddings for a list of entity IDs."""
    cache_path = MODEL_DIR / f"{cache_key}_emb_dict.pkl"
    if cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    texts = []
    valid_ids = []
    for eid in entity_ids:
        rec = norm_dict.get(eid, {})
        text = (rec.get("name_norm", "") + " " + rec.get("addr_norm", "")).strip()
        texts.append(text)
        valid_ids.append(eid)

    logger.info(f"Computing embeddings for {len(texts)} records ...")
    from sentence_transformers import SentenceTransformer
    from blocking import EMBED_MODEL, EMBED_BATCH
    m = model or SentenceTransformer(EMBED_MODEL, device="cuda:0")
    embs = m.encode(texts, batch_size=EMBED_BATCH, show_progress_bar=True,
                    normalize_embeddings=True, convert_to_numpy=True)

    emb_dict = dict(zip(valid_ids, embs))
    with open(cache_path, "wb") as f:
        pickle.dump(emb_dict, f)
    return emb_dict


def build_training_pairs(
    candidates_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    max_negatives_per_positive: int = 10,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create labelled pairs from candidates + ground truth.
    Returns (pairs_df, labels array).
    """
    # Parse ground truth
    gt_set: Set[Tuple[str, str]] = set()
    for _, row in ground_truth_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = str(row.get("matched_entity_ids", "") or "")
        for mid in matched.split(","):
            mid = mid.strip()
            if mid:
                gt_set.add((s1_id, mid))

    # Group candidates by S1
    s1_groups = candidates_df.groupby("source1_entity_id")

    pos_rows = []
    neg_rows = []

    for s1_id, group in s1_groups:
        cand_ids = list(group["candidate_entity_id"])
        pos_ids = [c for c in cand_ids if (s1_id, c) in gt_set]
        neg_ids = [c for c in cand_ids if (s1_id, c) not in gt_set]

        # Positives: all
        for c in pos_ids:
            pos_rows.append({"source1_entity_id": s1_id, "candidate_entity_id": c, "label": 1})

        # Hard negatives: up to N per positive
        n_neg = min(len(neg_ids), max(max_negatives_per_positive * len(pos_ids), 5))
        random.shuffle(neg_ids)
        for c in neg_ids[:n_neg]:
            neg_rows.append({"source1_entity_id": s1_id, "candidate_entity_id": c, "label": 0})

    all_rows = pos_rows + neg_rows
    pairs_df = pd.DataFrame(all_rows)
    labels = pairs_df["label"].values
    pairs_df = pairs_df.drop(columns=["label"])

    logger.info(
        f"Training pairs: {len(pos_rows)} positives + {len(neg_rows)} hard negatives "
        f"= {len(all_rows)} total"
    )
    return pairs_df, labels


def macro_f05(y_true: np.ndarray, y_pred: np.ndarray,
              groups: np.ndarray) -> float:
    """Compute macro-averaged F0.5 score per S1 entity group."""
    unique_groups = np.unique(groups)
    scores = []
    for g in unique_groups:
        mask = groups == g
        yt = y_true[mask]
        yp = y_pred[mask]
        tp = np.sum((yp == 1) & (yt == 1))
        fp = np.sum((yp == 1) & (yt == 0))
        fn = np.sum((yp == 0) & (yt == 1))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        if prec + rec == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
        scores.append(f05)
    return float(np.mean(scores))


def find_best_threshold(probs: np.ndarray, y_true: np.ndarray,
                         groups: np.ndarray) -> Tuple[float, float]:
    """Grid-search threshold on validation set for best macro F0.5."""
    best_thresh = 0.5
    best_score = 0.0
    for thresh in np.arange(0.2, 0.95, 0.02):
        preds = (probs >= thresh).astype(int)
        score = macro_f05(y_true, preds, groups)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh, best_score


def get_feature_cols(feat_df: pd.DataFrame) -> List[str]:
    """Return feature column names (exclude ID columns)."""
    exclude = {"source1_entity_id", "candidate_entity_id", "label"}
    return [c for c in feat_df.columns if c not in exclude]


# ─── Main training routine ────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Amazon MLC 2026 – Training Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load & normalise ───────────────────────────────────────────
    s1_norm_df, s1_norm_dict = load_and_normalize(DATA_DIR / "train_source1.tsv", "S1")
    s2_norm_df, s2_norm_dict = load_and_normalize(DATA_DIR / "train_source2.tsv", "S2")
    s3_norm_df, s3_norm_dict = load_and_normalize(DATA_DIR / "train_source3.tsv", "S3")

    all_norm_dict = {**s1_norm_dict, **s2_norm_dict, **s3_norm_dict}

    ground_truth = pd.read_csv(
        DATA_DIR / "train_ground_truth.tsv", sep="\t", dtype=str
    ).fillna("")

    # ── Step 2: Blocking ───────────────────────────────────────────────────
    cand_cache_path = MODEL_DIR / "train_candidates.parquet"
    if cand_cache_path.exists():
        logger.info("Loading cached training candidates ...")
        candidates_df = pd.read_parquet(cand_cache_path)
    else:
        logger.info("Generating training candidates ...")
        candidates_df = generate_candidates(
            s1_norm_df, s2_norm_df, s3_norm_df,
            use_embeddings=True, split_name="train"
        )
        candidates_df.to_parquet(cand_cache_path, index=False)
        logger.info(f"Saved candidates to {cand_cache_path}")

    # Evaluate blocking recall
    blocking_stats = evaluate_blocking(candidates_df, ground_truth)
    logger.info(f"\nBlocking stats: {blocking_stats}")

    # ── Step 3: Embeddings for features ───────────────────────────────────
    from sentence_transformers import SentenceTransformer
    from blocking import EMBED_MODEL
    embed_model = SentenceTransformer(EMBED_MODEL, device="cuda:0")

    all_ids_in_cands = (
        list(candidates_df["source1_entity_id"].unique())
        + list(candidates_df["candidate_entity_id"].unique())
    )
    emb_dict = compute_embeddings(
        all_norm_dict, all_ids_in_cands, embed_model, "train_all"
    )

    # ── Step 4: Build feature matrix ──────────────────────────────────────
    feat_cache_path = MODEL_DIR / "train_features.parquet"
    label_cache_path = MODEL_DIR / "train_labels.npy"

    if feat_cache_path.exists() and label_cache_path.exists():
        logger.info("Loading cached features and labels ...")
        feat_df = pd.read_parquet(feat_cache_path)
        labels = np.load(label_cache_path)
        # Guard: if shapes don't match the cache is stale — regenerate
        if len(labels) != len(feat_df):
            logger.warning(
                f"Cache shape mismatch: labels={len(labels)}, feat_df={len(feat_df)}. "
                "Regenerating features from scratch."
            )
            feat_cache_path.unlink(missing_ok=True)
            label_cache_path.unlink(missing_ok=True)
            train_pairs, labels = build_training_pairs(
                candidates_df, ground_truth, max_negatives_per_positive=10
            )
            logger.info("Building feature matrix ...")
            feat_df = build_feature_matrix_chunked(train_pairs, all_norm_dict, emb_dict)
            feat_df.to_parquet(feat_cache_path, index=False)
            np.save(label_cache_path, labels)
            logger.info(f"Saved features ({feat_df.shape}) and labels")
    else:
        # Build labelled pairs first
        train_pairs, labels = build_training_pairs(
            candidates_df, ground_truth, max_negatives_per_positive=10
        )
        logger.info("Building feature matrix ...")
        feat_df = build_feature_matrix_chunked(train_pairs, all_norm_dict, emb_dict)
        feat_df.to_parquet(feat_cache_path, index=False)
        np.save(label_cache_path, labels)
        logger.info(f"Saved features ({feat_df.shape}) and labels")

    feature_cols = get_feature_cols(feat_df)
    X = feat_df[feature_cols].values.astype(np.float32)
    y = labels
    groups = feat_df["source1_entity_id"].values

    logger.info(f"\nFeature matrix: {X.shape}, positives: {y.sum()}, negatives: {(y==0).sum()}")

    # ── Step 5: Group-split cross-validation ──────────────────────────────
    logger.info("\nGroup-split CV ...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    groups_val = groups[val_idx]

    # ── Step 6: Train LightGBM ─────────────────────────────────────────────
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info(f"\nTraining LightGBM (pos_weight={pos_weight:.2f}) ...")

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "scale_pos_weight": pos_weight,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )

    # ── Step 7: Threshold tuning ──────────────────────────────────────────
    val_probs = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f05 = find_best_threshold(val_probs, y_val, groups_val)
    logger.info(f"\nBest threshold: {best_thresh:.3f}  →  val F0.5: {best_f05:.4f}")

    # ── Step 8: Save model ────────────────────────────────────────────────
    model_path = MODEL_DIR / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "threshold": float(best_thresh),
        "val_f05": float(best_f05),
        "feature_cols": feature_cols,
        "blocking_stats": {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in blocking_stats.items()},
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\n✓ Model saved to {model_path}")
    logger.info(f"✓ Meta saved to {MODEL_DIR / 'model_meta.json'}")
    logger.info(f"\n  Blocking recall: {blocking_stats['pair_completeness_recall']:.4f}")
    logger.info(f"  Val F0.5:        {best_f05:.4f}")
    logger.info(f"  Decision thresh: {best_thresh:.3f}")

    # Feature importance
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    logger.info("\nTop 15 features:")
    logger.info(importances.head(15).to_string(index=False))
    importances.to_csv(MODEL_DIR / "feature_importances.csv", index=False)


if __name__ == "__main__":
    main()
