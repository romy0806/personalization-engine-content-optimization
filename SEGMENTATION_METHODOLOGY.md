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
- Freeze the train-fitted preprocessing, selected cluster count, model parameters, and persona names
- Apply that frozen pipeline to MIND dev without refitting
- Compare train/dev segment shares and feature distributions using Population Stability Index (PSI)
- Report validation separation and assignment-strength diagnostics

## Run with MIND

```bash
python -m scripts.run_segmentation \
  --train-behaviors data/MINDsmall_train/behaviors.tsv \
  --train-news data/MINDsmall_train/news.tsv \
  --validation-behaviors data/MINDsmall_dev/behaviors.tsv \
  --validation-news data/MINDsmall_dev/news.tsv
```

The runner deliberately has no synthetic-data fallback. It exports:

- `train_user_features.csv`
- `validation_user_features.csv`
- `train_segment_assignments.csv`
- `train_segment_profiles.csv`
- `segmentation_model_comparison.csv`
- `validation_segment_assignments.csv`
- `validation_segment_profiles.csv`
- `train_validation_segment_comparison.csv`
- `validation_metrics.csv`
- `validation_feature_drift.csv`

For PSI, values below 0.10 are commonly treated as limited drift, 0.10–0.25 as a review
signal, and above 0.25 as material drift. These are operational guides rather than universal
statistical cutoffs. Small MIND-format fixtures are used only by automated tests. The repository
does not redistribute the Microsoft dataset.
