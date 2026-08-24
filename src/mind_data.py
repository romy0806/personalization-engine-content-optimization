"""Load Microsoft MIND files and engineer user-level segmentation features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BEHAVIOR_COLUMNS = ("impression_id", "user_id", "timestamp", "history", "impressions")
NEWS_COLUMNS = (
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
)


def read_mind_behaviors(path: str | Path) -> pd.DataFrame:
    """Read a MIND behaviors.tsv file using the published schema."""
    data = pd.read_csv(path, sep="\t", names=BEHAVIOR_COLUMNS, dtype=str)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if data["user_id"].isna().any() or data["timestamp"].isna().all():
        raise ValueError("Invalid MIND behaviors file: user IDs and timestamps are required")
    return data


def read_mind_news(path: str | Path) -> pd.DataFrame:
    """Read a MIND news.tsv file using the published schema."""
    data = pd.read_csv(path, sep="\t", names=NEWS_COLUMNS, dtype=str)
    if data["news_id"].isna().any():
        raise ValueError("Invalid MIND news file: news IDs are required")
    return data


def _history_ids(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return str(value).split()


def _impression_pairs(value: object) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    if pd.isna(value):
        return pairs
    for token in str(value).split():
        news_id, separator, label = token.rpartition("-")
        if not separator or label not in {"0", "1"}:
            continue
        pairs.append((news_id, int(label)))
    return pairs


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).fillna(0.0)


def build_mind_user_features(
    behaviors: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate MIND interactions into explainable user-level features.

    Features use only fields supplied by MIND. They are calculated across the
    provided observation window; the maximum event timestamp is the recency
    reference date.
    """
    missing_behaviors = sorted(set(BEHAVIOR_COLUMNS) - set(behaviors.columns))
    missing_news = sorted({"news_id", "category", "subcategory"} - set(news.columns))
    if missing_behaviors or missing_news:
        missing = missing_behaviors + missing_news
        raise ValueError(f"Missing required MIND columns: {', '.join(missing)}")

    events = behaviors.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    events = events.dropna(subset=["user_id", "timestamp"])
    if events.empty:
        raise ValueError("No valid MIND behavior records were found")

    category_map = news.drop_duplicates("news_id").set_index("news_id")["category"].to_dict()
    subcategory_map = news.drop_duplicates("news_id").set_index("news_id")["subcategory"].to_dict()
    reference_time = events["timestamp"].max()
    rows: list[dict[str, float | int | str]] = []

    for user_id, user_events in events.groupby("user_id", sort=False):
        user_events = user_events.sort_values("timestamp")
        latest_history = _history_ids(user_events.iloc[-1]["history"])
        impression_lists = [_impression_pairs(value) for value in user_events["impressions"]]
        impression_pairs = [pair for impressions in impression_lists for pair in impressions]
        exposed_ids = [news_id for news_id, _ in impression_pairs]
        clicked_ids = [news_id for news_id, label in impression_pairs if label == 1]
        consumed_ids = latest_history + clicked_ids
        categories = [category_map.get(news_id) for news_id in consumed_ids]
        categories = [value for value in categories if pd.notna(value)]
        subcategories = [subcategory_map.get(news_id) for news_id in consumed_ids]
        subcategories = [value for value in subcategories if pd.notna(value)]
        category_counts = pd.Series(categories, dtype="object").value_counts()
        dominant_share = (
            float(category_counts.iloc[0] / category_counts.sum())
            if not category_counts.empty
            else 0.0
        )
        impressions_total = len(exposed_ids)
        clicks_total = len(clicked_ids)
        rows.append(
            {
                "user_id": str(user_id),
                "recency_days": float(
                    (reference_time - user_events["timestamp"].max()).total_seconds() / 86400
                ),
                "sessions": len(user_events),
                "active_days": int(user_events["timestamp"].dt.normalize().nunique()),
                "history_length": len(latest_history),
                "impressions_total": int(impressions_total),
                "clicks_total": int(clicks_total),
                "click_through_rate": float(clicks_total / impressions_total)
                if impressions_total
                else 0.0,
                "avg_impression_slate_size": float(impressions_total / len(user_events)),
                "category_diversity": len(set(categories)),
                "subcategory_diversity": len(set(subcategories)),
                "dominant_category_share": dominant_share,
            }
        )

    features = pd.DataFrame(rows)
    features["clicks_per_session"] = _safe_ratio(features["clicks_total"], features["sessions"])
    return features


def load_mind_user_features(
    behaviors_path: str | Path,
    news_path: str | Path,
) -> pd.DataFrame:
    """Read MIND TSV files and return the segmentation feature table."""
    return build_mind_user_features(
        read_mind_behaviors(behaviors_path),
        read_mind_news(news_path),
    )
