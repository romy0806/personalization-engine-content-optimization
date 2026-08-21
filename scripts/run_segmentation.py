"""Run the validated segmentation pipeline and export reviewable evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.segmentation import SegmentationConfig, fit_segmentation
from src.synthetic_data import generate_behavioral_users


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional user-level behavioral CSV")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--demo-users-per-segment", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    users = (
        pd.read_csv(args.input)
        if args.input
        else generate_behavioral_users(args.demo_users_per_segment)
    )
    result = fit_segmentation(users, SegmentationConfig())

    args.output.mkdir(parents=True, exist_ok=True)
    result.assignments.to_csv(args.output / "segment_assignments.csv", index=False)
    result.profiles.to_csv(args.output / "segment_profiles.csv", index=False)
    result.candidates.to_csv(args.output / "segmentation_model_comparison.csv", index=False)

    selected = result.candidates.iloc[0]
    print(
        "Selected "
        f"{result.selected_algorithm} with {result.selected_clusters} segments; "
        f"silhouette={selected['silhouette']:.3f}, "
        f"stability_ari={selected['stability_ari']:.3f}."
    )


if __name__ == "__main__":
    main()
