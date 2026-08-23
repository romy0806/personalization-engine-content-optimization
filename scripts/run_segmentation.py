"""Build Microsoft MIND user features, fit segmentation, and export evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.mind_data import load_mind_user_features
from src.segmentation import SegmentationConfig, fit_segmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--behaviors", type=Path, required=True, help="MIND behaviors.tsv"
    )
    parser.add_argument("--news", type=Path, required=True, help="MIND news.tsv")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    users = load_mind_user_features(args.behaviors, args.news)
    result = fit_segmentation(users, SegmentationConfig())
    args.output.mkdir(parents=True, exist_ok=True)
    users.to_csv(args.output / "mind_user_features.csv", index=False)
    result.assignments.to_csv(args.output / "segment_assignments.csv", index=False)
    result.profiles.to_csv(args.output / "segment_profiles.csv", index=False)
    result.candidates.to_csv(
        args.output / "segmentation_model_comparison.csv", index=False
    )
    selected = result.candidates.iloc[0]
    print(
        f"Selected {result.selected_algorithm} with {result.selected_clusters} segments; "
        f"silhouette={selected['silhouette']:.3f}, stability_ari={selected['stability_ari']:.3f}."
    )


if __name__ == "__main__":
    main()
