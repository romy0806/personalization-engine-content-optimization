"""
Modeling utilities for content scoring and user segmentation.
"""

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


def train_user_segments(features, n_clusters: int = 5, random_state: int = 42):
    """Cluster users into behavioral segments."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(features)
    return model, labels


def train_engagement_model(X, y, random_state: int = 42):
    """Train a simple engagement prediction model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=random_state,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return model, auc
