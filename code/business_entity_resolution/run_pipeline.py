"""
run_pipeline.py
===============
Single entry-point convenience script.

Run from the student_resource/ directory:
    python code/business_entity_resolution/run_pipeline.py [--train] [--predict] [--evaluate]

Flags:
  --train    Run training pipeline (blocking + features + LightGBM)
  --predict  Run prediction pipeline (generates output/*.tsv)
  --evaluate Evaluate predictions against training ground truth
  --all      Run all stages in order (default if no flag given)
"""

import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"


def run(script: str):
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(SRC / script)],
        check=True,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline runner")
    parser.add_argument("--train", action="store_true", help="Run training stage")
    parser.add_argument("--predict", action="store_true", help="Run prediction stage")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate predictions")
    parser.add_argument("--all", action="store_true", help="Run all stages (default)")
    args = parser.parse_args()

    run_all = args.all or not any([args.train, args.predict, args.evaluate])

    if run_all or args.train:
        run("train.py")

    if run_all or args.predict:
        run("predict.py")

    if run_all or args.evaluate:
        run("evaluate.py")

    print("\n✓ Pipeline complete.")
    print("  Validate with:")
    print("    python utils/validate_submission.py \\")
    print("        --matching output/matching_results.tsv \\")
    print("        --candidate output/candidate_pairs.tsv \\")
    print("        --test-dir dataset/test")


if __name__ == "__main__":
    main()
