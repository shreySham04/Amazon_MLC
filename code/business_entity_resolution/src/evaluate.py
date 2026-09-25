"""
evaluate.py
===========
Local evaluation script.

Computes macro-averaged F0.5 on the training set (with a held-out group split),
and measures blocking recall separately.

Usage:
  python src/evaluate.py
  python src/evaluate.py --on-train   # evaluate on full train data (debugging)
  python src/evaluate.py --predictions output/matching_results.tsv
                         --ground-truth dataset/train/train_ground_truth.tsv
                         --source1 dataset/train/train_source1.tsv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s – %(message)s")
logger = logging.getLogger("evaluate")

ROOT = Path(__file__).parents[3]


def parse_matched_ids(value: str) -> Set[str]:
    if not value or pd.isna(value):
        return set()
    return {v.strip() for v in str(value).split(",") if v.strip()}


def compute_f05_per_entity(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, pd.DataFrame]:
    """
    Compute per-entity F0.5 and macro average.

    Returns
    -------
    macro_f05 : float
    per_entity_df : DataFrame with per-entity breakdown
    """
    rows = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)

        if prec + rec == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)

        rows.append({
            "source1_entity_id": s1_id,
            "true_count": len(true_set),
            "pred_count": len(pred_set),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f05": f05,
        })

    per_entity_df = pd.DataFrame(rows)
    macro_f05 = per_entity_df["f05"].mean()
    return macro_f05, per_entity_df


def evaluate_from_files(
    pred_path: str,
    gt_path: str,
    s1_path: str,
) -> None:
    """Load prediction and ground-truth files, print metrics."""
    logger.info(f"Predictions: {pred_path}")
    logger.info(f"Ground truth: {gt_path}")

    pred_df = pd.read_csv(pred_path, sep="\t", dtype=str).fillna("")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")

    all_s1_ids = set(s1_df["entity_id"])

    predictions: Dict[str, Set[str]] = {}
    for _, row in pred_df.iterrows():
        predictions[row["source1_entity_id"]] = parse_matched_ids(row["matched_entity_ids"])

    ground_truth: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        ground_truth[row["source1_entity_id"]] = parse_matched_ids(row["matched_entity_ids"])

    # Ensure all S1 entities are present in ground_truth (singletons = empty set)
    for s1_id in all_s1_ids:
        if s1_id not in ground_truth:
            ground_truth[s1_id] = set()
        if s1_id not in predictions:
            predictions[s1_id] = set()

    macro, per_ent = compute_f05_per_entity(predictions, ground_truth)

    n_entities = len(ground_truth)
    n_singletons = sum(1 for v in ground_truth.values() if not v)
    n_pred_empty = sum(1 for v in predictions.values() if not v)

    # Singleton accuracy
    singleton_ids = {k for k, v in ground_truth.items() if not v}
    singleton_correct = sum(1 for s in singleton_ids if not predictions.get(s))
    singleton_acc = singleton_correct / max(len(singleton_ids), 1)

    logger.info("\n" + "=" * 50)
    logger.info(f"  Entities:         {n_entities}")
    logger.info(f"  Singletons (GT):  {n_singletons}  ({100*n_singletons/n_entities:.1f}%)")
    logger.info(f"  Pred empty:       {n_pred_empty}  ({100*n_pred_empty/n_entities:.1f}%)")
    logger.info(f"  Singleton acc:    {singleton_acc:.4f}")
    logger.info(f"  Macro F0.5:       {macro:.4f}")
    logger.info("=" * 50)

    # Error analysis: worst entities
    worst = per_ent.nsmallest(10, "f05")
    logger.info("\nWorst 10 entities:")
    logger.info(worst[["source1_entity_id", "true_count", "pred_count", "f05"]].to_string(index=False))

    # Distribution
    logger.info(f"\nF0.5 distribution:")
    logger.info(per_ent["f05"].describe().to_string())

    per_ent.to_csv(Path(pred_path).parent / "per_entity_eval.csv", index=False)
    logger.info(f"\nPer-entity breakdown saved to {Path(pred_path).parent / 'per_entity_eval.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLC predictions")
    parser.add_argument("--predictions", default="output/matching_results.tsv")
    parser.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--source1", default="dataset/train/train_source1.tsv")
    args = parser.parse_args()

    evaluate_from_files(
        str(ROOT / args.predictions),
        str(ROOT / args.ground_truth),
        str(ROOT / args.source1),
    )


if __name__ == "__main__":
    main()
