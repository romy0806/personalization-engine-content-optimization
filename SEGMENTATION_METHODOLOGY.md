# Personalization Engine 2.0: MIND-Based Validated Segmentation

This phase upgrades the original Microsoft MIND analysis with a reproducible user-level feature pipeline and evidence-based cluster selection. The analytical source remains MIND `behaviors.tsv` and `news.tsv`.

## MIND-derived features

- Recency relative to the latest timestamp in the supplied observation window
- Sessions and active days
- Reading-history length
- Total recommendation impressions and clicks
- Click-through rate and clicks per session
- Average impression-slate size
- Category and subcategory diversity
- Dominant-category share

These definitions are limited to fields provided by MIND. They do not claim subscription status, conversion, purchase intent, or other outcomes that MIND does not contain.

## Model-selection workflow

- Validate the MIND schemas and user population
- Impute missing values, cap extreme observations, and apply robust scaling
- Compare K-Means and Gaussian Mixture candidates across 2–8 segments
- Evaluate Silhouette, Davies–Bouldin, Calinski–Harabasz, and bootstrap Adjusted Rand Index
- Reject extremely small or dominant clusters
- Generate descriptive segment names from relative behavioral evidence
- Report relative assignment strength as a diagnostic, not a calibrated probability

## Run with MIND

```bash
python -m scripts.run_segmentation \
  --behaviors data/MINDsmall_train/behaviors.tsv \
  --news data/MINDsmall_train/news.tsv
```

The runner deliberately has no synthetic-data fallback. It exports:

- `mind_user_features.csv`
- `segment_assignments.csv`
- `segment_profiles.csv`
- `segmentation_model_comparison.csv`

Small MIND-format fixtures are used only by automated tests. The repository does not redistribute the Microsoft dataset.
