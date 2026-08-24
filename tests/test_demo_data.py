from pathlib import Path

import pandas as pd
import pytest

DEMO_DATA = Path(__file__).resolve().parents[1] / "app" / "demo_data"


def test_demo_snapshot_reconciles_user_populations() -> None:
    segments = pd.read_csv(DEMO_DATA / "segment_comparison.csv")

    assert segments["train_users"].sum() == 50_000
    assert segments["validation_users"].sum() == 50_000
    assert segments["train_share"].sum() == pytest.approx(1.0)
    assert segments["validation_share"].sum() == pytest.approx(1.0)


def test_demo_snapshot_matches_selected_model() -> None:
    models = pd.read_csv(DEMO_DATA / "model_comparison.csv")
    selected = models.iloc[0]

    assert selected["algorithm"] == "kmeans"
    assert selected["clusters"] == 5
    assert bool(selected["business_viable"])
    assert selected["stability_ari"] > 0.95


def test_demo_snapshot_has_one_profile_per_persona() -> None:
    segments = pd.read_csv(DEMO_DATA / "segment_comparison.csv")
    profiles = pd.read_csv(DEMO_DATA / "validation_profiles.csv")

    assert profiles["segment_name"].is_unique
    assert set(profiles["segment_name"]) == set(segments["segment_name"])
