# Personalization Engine for Content Optimization

![Hero Banner](hero_banner.png)

## Objective

Build and evaluate a content-personalization framework using the Microsoft MIND news recommendation dataset. The project tests whether behavioral user segmentation improves click prediction and provides a validated path toward recommendation ranking.

## Analytical approach

1. Parse MIND `behaviors.tsv` and `news.tsv`.
2. Engineer user-level interaction, recency, click, exposure, and topic-affinity features.
3. Compare K-Means and Gaussian Mixture segmentation candidates.
4. Validate separation, bootstrap stability, and segment-size viability.
5. Compare a content-only baseline with a model augmented by user behavior/segments.

The original analysis reported AUC of 0.57 for the baseline and 0.69 for the enhanced model. Those historical results should be reproduced through the versioned pipeline before being treated as a release benchmark.

## Run segmentation

Download MINDsmall and place its files under `data/MINDsmall_train/` and `data/MINDsmall_dev/`. The dataset is not redistributed in this repository.

```bash
python -m pip install -r requirements-dev.txt
python -m scripts.run_segmentation \
  --behaviors data/MINDsmall_train/behaviors.tsv \
  --news data/MINDsmall_train/news.tsv
python -m pytest -q
```

## What is validated

- MIND schema parsing and user-level feature engineering
- Robust preprocessing and outlier handling
- K-Means versus Gaussian Mixture model selection
- Silhouette, Davies–Bouldin, Calinski–Harabasz, and bootstrap ARI evidence
- Cluster-size guardrails and assignment-strength diagnostics
- Python 3.11 and 3.12 CI

## Current boundaries

The Streamlit screen remains a static interface prototype. Recommendation ranking, React/FastAPI architecture, live integrations, and production monitoring are separate future phases. Test fixtures mimic the MIND schema but are not used as analytical evidence.

## Technology

Python, Pandas, NumPy, scikit-learn, SciPy, Streamlit, pytest, Ruff, and GitHub Actions.
