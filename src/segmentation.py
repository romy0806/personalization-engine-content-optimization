"""Validated behavioral segmentation for Microsoft MIND user features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

DEFAULT_FEATURES = (
    "recency_days",
    "sessions",
    "active_days",
    "history_length",
    "impressions_total",
    "click_through_rate",
    "clicks_per_session",
    "avg_impression_slate_size",
    "category_diversity",
    "subcategory_diversity",
    "dominant_category_share",
)


@dataclass(frozen=True)
class SegmentationConfig:
    features: tuple[str, ...] = DEFAULT_FEATURES
    cluster_range: tuple[int, ...] = tuple(range(2, 9))
    algorithms: tuple[str, ...] = ("kmeans", "gaussian_mixture")
    min_cluster_share: float = 0.05
    max_cluster_share: float = 0.60
    bootstrap_iterations: int = 5
    bootstrap_fraction: float = 0.80
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    random_state: int = 42


@dataclass
class SegmentationResult:
    assignments: pd.DataFrame
    profiles: pd.DataFrame
    candidates: pd.DataFrame
    selected_algorithm: str
    selected_clusters: int
    model: Any
    preprocessor: BehavioralPreprocessor


class BehavioralPreprocessor:
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.medians_: pd.Series | None = None
        self.lower_: pd.Series | None = None
        self.upper_: pd.Series | None = None
        self.scaler_: RobustScaler | None = None

    def fit(self, data: pd.DataFrame) -> BehavioralPreprocessor:
        numeric = _validated_numeric_features(data, self.config)
        self.medians_ = numeric.median()
        filled = numeric.fillna(self.medians_)
        self.lower_ = filled.quantile(self.config.lower_quantile)
        self.upper_ = filled.quantile(self.config.upper_quantile)
        clipped = filled.clip(self.lower_, self.upper_, axis="columns")
        self.scaler_ = RobustScaler().fit(clipped)
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if any(
            value is None
            for value in (self.medians_, self.lower_, self.upper_, self.scaler_)
        ):
            raise RuntimeError("BehavioralPreprocessor must be fitted before transform")
        numeric = _validated_numeric_features(data, self.config)
        filled = numeric.fillna(self.medians_)
        clipped = filled.clip(self.lower_, self.upper_, axis="columns")
        return self.scaler_.transform(clipped)

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.fit(data).transform(data)


def _validated_numeric_features(
    data: pd.DataFrame, config: SegmentationConfig
) -> pd.DataFrame:
    missing = sorted(set(config.features) - set(data.columns))
    if missing:
        raise ValueError(
            f"Missing required segmentation features: {', '.join(missing)}"
        )
    if len(data) < 60:
        raise ValueError("Segmentation requires at least 60 users")
    if not 0 < config.min_cluster_share < config.max_cluster_share < 1:
        raise ValueError("Cluster-share guardrails must satisfy 0 < min < max < 1")
    numeric = data.loc[:, config.features].apply(pd.to_numeric, errors="coerce")
    all_missing = numeric.columns[numeric.isna().all()].tolist()
    if all_missing:
        raise ValueError(
            f"Features contain no usable numeric values: {', '.join(all_missing)}"
        )
    return numeric.replace([np.inf, -np.inf], np.nan)


def _build_model(algorithm: str, clusters: int, random_state: int):
    if algorithm == "kmeans":
        return KMeans(n_clusters=clusters, n_init=20, random_state=random_state)
    if algorithm == "gaussian_mixture":
        return GaussianMixture(
            n_components=clusters,
            covariance_type="diag",
            n_init=3,
            reg_covar=1e-5,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported segmentation algorithm: {algorithm}")


def _fit_predict(model, features: np.ndarray) -> np.ndarray:
    if isinstance(model, GaussianMixture):
        model.fit(features)
        return model.predict(features)
    return model.fit_predict(features)


def _assignment_strength(model, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if isinstance(model, GaussianMixture):
        return model.predict_proba(features).max(axis=1)
    distances = np.maximum(model.transform(features), 1e-12)
    inverse = 1 / distances
    relative = inverse / inverse.sum(axis=1, keepdims=True)
    return relative[np.arange(len(features)), labels]


def _bootstrap_stability(model, features, reference_labels, config) -> float:
    rng = np.random.default_rng(config.random_state)
    scores: list[float] = []
    size = max(30, int(len(features) * config.bootstrap_fraction))
    for iteration in range(config.bootstrap_iterations):
        indices = rng.choice(len(features), size=size, replace=True)
        candidate = clone(model)
        if hasattr(candidate, "random_state"):
            candidate.set_params(random_state=config.random_state + iteration + 1)
        candidate.fit(features[indices])
        labels = candidate.predict(features)
        if len(np.unique(labels)) >= 2:
            scores.append(adjusted_rand_score(reference_labels, labels))
    return float(np.mean(scores)) if scores else 0.0


def _minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    spread = series.max() - series.min()
    normalized = (
        pd.Series(0.5, index=series.index)
        if spread == 0
        else (series - series.min()) / spread
    )
    return normalized if higher_is_better else 1 - normalized


def _persona_name(z: pd.Series) -> str:
    if z["recency_days"] > 0.6 and z["sessions"] < -0.35:
        return "Dormant readers"
    if z["click_through_rate"] > 0.5 and z["clicks_per_session"] > 0.35:
        return "Highly responsive readers"
    if z["dominant_category_share"] > 0.55 and z["history_length"] > 0.2:
        return "Loyal topic specialists"
    if z["category_diversity"] > 0.5:
        return "Cross-topic explorers"
    if z["sessions"] < -0.35:
        return "Light readers"
    strongest = z.sort_values(ascending=False).index[0].replace("_", " ").title()
    return f"{strongest} segment"


def _profiles(
    data: pd.DataFrame, labels: np.ndarray, config: SegmentationConfig
) -> pd.DataFrame:
    numeric = data.loc[:, config.features].apply(pd.to_numeric, errors="coerce")
    means = numeric.assign(segment_id=labels).groupby("segment_id").mean()
    counts = pd.Series(labels).value_counts().sort_index()
    z_scores = means.sub(numeric.mean()).div(numeric.std(ddof=0).replace(0, 1))
    rows = []
    for segment_id in means.index:
        row = {
            "segment_id": int(segment_id),
            "segment_name": _persona_name(z_scores.loc[segment_id]),
            "users": int(counts.loc[segment_id]),
            "user_share": float(counts.loc[segment_id] / len(data)),
            "defining_features": ", ".join(
                z_scores.loc[segment_id].sort_values(ascending=False).head(3).index
            ),
        }
        row.update(means.loc[segment_id].astype(float).to_dict())
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("user_share", ascending=False)
        .reset_index(drop=True)
    )


def fit_segmentation(
    data: pd.DataFrame, config: SegmentationConfig | None = None
) -> SegmentationResult:
    config = config or SegmentationConfig()
    preprocessor = BehavioralPreprocessor(config)
    features = preprocessor.fit_transform(data)
    evidence = []
    fitted = {}
    for algorithm in config.algorithms:
        for clusters in config.cluster_range:
            if clusters >= len(data):
                continue
            model = _build_model(algorithm, clusters, config.random_state)
            labels = _fit_predict(model, features)
            if len(np.unique(labels)) < 2:
                continue
            shares = np.bincount(labels, minlength=clusters) / len(labels)
            viable = bool(
                shares.min() >= config.min_cluster_share
                and shares.max() <= config.max_cluster_share
            )
            fitted[(algorithm, clusters)] = (model, labels)
            evidence.append(
                {
                    "algorithm": algorithm,
                    "clusters": clusters,
                    "silhouette": float(silhouette_score(features, labels)),
                    "davies_bouldin": float(davies_bouldin_score(features, labels)),
                    "calinski_harabasz": float(
                        calinski_harabasz_score(features, labels)
                    ),
                    "stability_ari": _bootstrap_stability(
                        model, features, labels, config
                    ),
                    "smallest_cluster_share": float(shares.min()),
                    "largest_cluster_share": float(shares.max()),
                    "business_viable": viable,
                }
            )
    if not evidence:
        raise ValueError("No valid segmentation candidates could be fitted")
    candidates = pd.DataFrame(evidence)
    candidates["selection_score"] = (
        0.40 * _minmax(candidates["silhouette"])
        + 0.25 * _minmax(candidates["davies_bouldin"], False)
        + 0.15 * _minmax(np.log1p(candidates["calinski_harabasz"]))
        + 0.20 * _minmax(candidates["stability_ari"])
    )
    candidates.loc[~candidates["business_viable"], "selection_score"] -= 1
    candidates = candidates.sort_values("selection_score", ascending=False).reset_index(
        drop=True
    )
    selected = candidates.iloc[0]
    if not bool(selected["business_viable"]):
        raise ValueError("No segmentation candidate passed the cluster-size guardrails")
    key = (str(selected["algorithm"]), int(selected["clusters"]))
    model, labels = fitted[key]
    profiles = _profiles(data, labels, config)
    names = profiles.set_index("segment_id")["segment_name"].to_dict()
    user_ids = data["user_id"] if "user_id" in data else pd.Series(data.index)
    assignments = pd.DataFrame(
        {
            "user_id": user_ids.to_numpy(),
            "segment_id": labels,
            "segment_name": [names[int(label)] for label in labels],
            "assignment_strength": _assignment_strength(model, features, labels),
        }
    )
    return SegmentationResult(
        assignments, profiles, candidates, key[0], key[1], model, preprocessor
    )
