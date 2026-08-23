"""Validated, deployable behavioral segmentation methodology."""

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
    "sessions_30d",
    "articles_viewed_30d",
    "click_through_rate",
    "category_diversity",
    "repeat_visit_rate",
    "avg_session_depth",
    "high_intent_actions_30d",
)


@dataclass(frozen=True)
class SegmentationConfig:
    """Controls candidate selection and business viability guardrails."""

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
    """All fitted artifacts and evidence needed by an application layer."""

    assignments: pd.DataFrame
    profiles: pd.DataFrame
    candidates: pd.DataFrame
    selected_algorithm: str
    selected_clusters: int
    model: Any
    preprocessor: BehavioralPreprocessor


class BehavioralPreprocessor:
    """Fit reproducible clipping, missing-value and scaling parameters."""

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
        if any(value is None for value in (self.medians_, self.lower_, self.upper_, self.scaler_)):
            raise RuntimeError("BehavioralPreprocessor must be fitted before transform")
        numeric = _validated_numeric_features(data, self.config)
        filled = numeric.fillna(self.medians_)
        clipped = filled.clip(self.lower_, self.upper_, axis="columns")
        return self.scaler_.transform(clipped)

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.fit(data).transform(data)


def _validated_numeric_features(data: pd.DataFrame, config: SegmentationConfig) -> pd.DataFrame:
    missing = sorted(set(config.features) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required segmentation features: {', '.join(missing)}")
    if len(data) < 60:
        raise ValueError("Segmentation requires at least 60 users")
    if not 0 < config.min_cluster_share < config.max_cluster_share < 1:
        raise ValueError("Cluster-share guardrails must satisfy 0 < min < max < 1")

    numeric = data.loc[:, config.features].apply(pd.to_numeric, errors="coerce")
    all_missing = numeric.columns[numeric.isna().all()].tolist()
    if all_missing:
        raise ValueError(f"Features contain no usable numeric values: {', '.join(all_missing)}")
    return numeric.replace([np.inf, -np.inf], np.nan)


def _build_model(algorithm: str, clusters: int, random_state: int):
    if algorithm == "kmeans":
        return KMeans(
            n_clusters=clusters,
            n_init=20,
            random_state=random_state,
        )
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
    """Return relative assignment strength, not a calibrated probability."""
    if isinstance(model, GaussianMixture):
        return model.predict_proba(features).max(axis=1)

    distances = model.transform(features)
    safe_distances = np.maximum(distances, 1e-12)
    inverse = 1 / safe_distances
    relative_strength = inverse / inverse.sum(axis=1, keepdims=True)
    return relative_strength[np.arange(len(features)), labels]


def _bootstrap_stability(
    model,
    features: np.ndarray,
    reference_labels: np.ndarray,
    config: SegmentationConfig,
) -> float:
    rng = np.random.default_rng(config.random_state)
    scores: list[float] = []
    sample_size = max(30, int(len(features) * config.bootstrap_fraction))
    cluster_count = len(np.unique(reference_labels))

    for iteration in range(config.bootstrap_iterations):
        indices = rng.choice(len(features), size=sample_size, replace=True)
        if len(np.unique(indices)) <= cluster_count:
            continue

        candidate = clone(model)
        if hasattr(candidate, "random_state"):
            candidate.set_params(random_state=config.random_state + iteration + 1)

        candidate.fit(features[indices])
        candidate_labels = candidate.predict(features)
        if len(np.unique(candidate_labels)) < 2:
            continue
        scores.append(adjusted_rand_score(reference_labels, candidate_labels))

    return float(np.mean(scores)) if scores else 0.0


def _candidate_evidence(
    algorithm: str,
    clusters: int,
    model,
    features: np.ndarray,
    labels: np.ndarray,
    config: SegmentationConfig,
) -> dict[str, float | int | str | bool]:
    counts = np.bincount(labels, minlength=clusters)
    shares = counts / counts.sum()
    viable = bool(
        shares.min() >= config.min_cluster_share and shares.max() <= config.max_cluster_share
    )
    return {
        "algorithm": algorithm,
        "clusters": clusters,
        "silhouette": float(silhouette_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
        "stability_ari": _bootstrap_stability(model, features, labels, config),
        "smallest_cluster_share": float(shares.min()),
        "largest_cluster_share": float(shares.max()),
        "business_viable": viable,
    }


def _minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    spread = series.max() - series.min()
    normalized = (
        pd.Series(0.5, index=series.index) if spread == 0 else (series - series.min()) / spread
    )
    return normalized if higher_is_better else 1 - normalized


def _score_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    scored = candidates.copy()
    scored["selection_score"] = (
        0.40 * _minmax(scored["silhouette"])
        + 0.25 * _minmax(scored["davies_bouldin"], higher_is_better=False)
        + 0.15 * _minmax(np.log1p(scored["calinski_harabasz"]))
        + 0.20 * _minmax(scored["stability_ari"])
    )
    scored.loc[~scored["business_viable"], "selection_score"] -= 1
    return scored.sort_values("selection_score", ascending=False).reset_index(drop=True)


def _segment_profiles(
    data: pd.DataFrame, labels: np.ndarray, config: SegmentationConfig
) -> pd.DataFrame:
    numeric = data.loc[:, config.features].apply(pd.to_numeric, errors="coerce")
    population_mean = numeric.mean()
    population_std = numeric.std(ddof=0).replace(0, 1)
    working = numeric.assign(segment_id=labels)
    means = working.groupby("segment_id").mean()
    counts = working.groupby("segment_id").size()
    z_scores = means.sub(population_mean).div(population_std)

    rows: list[dict[str, Any]] = []
    for segment_id in means.index:
        defining = z_scores.loc[segment_id].sort_values(ascending=False).head(3)
        weak = z_scores.loc[segment_id].sort_values().head(2)
        name = _persona_name(z_scores.loc[segment_id])
        row: dict[str, Any] = {
            "segment_id": int(segment_id),
            "segment_name": name,
            "users": int(counts.loc[segment_id]),
            "user_share": float(counts.loc[segment_id] / len(data)),
            "defining_features": ", ".join(defining.index),
            "lower_features": ", ".join(weak.index),
        }
        row.update({feature: float(means.loc[segment_id, feature]) for feature in config.features})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("user_share", ascending=False).reset_index(drop=True)


def _persona_name(z_scores: pd.Series) -> str:
    """Assign transparent names from relative behavioral evidence."""
    if z_scores["recency_days"] > 0.75 and z_scores["sessions_30d"] < -0.50:
        return "At-risk users"
    if z_scores["high_intent_actions_30d"] > 0.65 and z_scores["click_through_rate"] > 0.35:
        return "High-intent researchers"
    if z_scores["repeat_visit_rate"] > 0.65 and z_scores["category_diversity"] < -0.25:
        return "Loyal category specialists"
    if z_scores["category_diversity"] > 0.55:
        return "Broad explorers"
    if z_scores["sessions_30d"] < -0.35:
        return "Casual visitors"
    strongest = z_scores.sort_values(ascending=False).index[0].replace("_", " ").title()
    return f"{strongest} segment"


def fit_segmentation(
    data: pd.DataFrame,
    config: SegmentationConfig | None = None,
) -> SegmentationResult:
    """Select, fit and explain a deployable behavioral segmentation model."""
    config = config or SegmentationConfig()
    preprocessor = BehavioralPreprocessor(config)
    features = preprocessor.fit_transform(data)

    evidence: list[dict[str, Any]] = []
    fitted: dict[tuple[str, int], tuple[Any, np.ndarray]] = {}
    for algorithm in config.algorithms:
        for clusters in config.cluster_range:
            if clusters >= len(data):
                continue
            model = _build_model(algorithm, clusters, config.random_state)
            labels = _fit_predict(model, features)
            if len(np.unique(labels)) < 2:
                continue
            fitted[(algorithm, clusters)] = (model, labels)
            evidence.append(
                _candidate_evidence(algorithm, clusters, model, features, labels, config)
            )

    if not evidence:
        raise ValueError("No valid segmentation candidates could be fitted")

    candidates = _score_candidates(pd.DataFrame(evidence))
    selected = candidates.iloc[0]
    if not bool(selected["business_viable"]):
        raise ValueError("No segmentation candidate passed the cluster-size guardrails")

    key = (str(selected["algorithm"]), int(selected["clusters"]))
    model, labels = fitted[key]
    assignment_strength = _assignment_strength(model, features, labels)
    profiles = _segment_profiles(data, labels, config)
    name_map = profiles.set_index("segment_id")["segment_name"].to_dict()

    user_ids = (
        data["user_id"] if "user_id" in data.columns else pd.Series(data.index, index=data.index)
    )
    assignments = pd.DataFrame(
        {
            "user_id": user_ids.to_numpy(),
            "segment_id": labels,
            "segment_name": [name_map[int(label)] for label in labels],
            "assignment_strength": assignment_strength,
        }
    )
    return SegmentationResult(
        assignments=assignments,
        profiles=profiles,
        candidates=candidates,
        selected_algorithm=key[0],
        selected_clusters=key[1],
        model=model,
        preprocessor=preprocessor,
    )
