# Personalization Engine 2.0: Validated Segmentation

This Phase 1 package replaces fixed-count user clustering with a reproducible model-selection and validation workflow. It is intentionally independent of the current static Streamlit demo so the methodology can be validated before the React and FastAPI application is built.

## What it adds

- User-level behavioral feature contract
- Missing-value handling, outlier capping and robust scaling
- K-Means and Gaussian Mixture comparison across 2–8 segments
- Silhouette, Davies–Bouldin and Calinski–Harabasz evidence
- Bootstrap stability using Adjusted Rand Index
- Minimum and maximum segment-size guardrails
- Membership confidence for every user
- Evidence-based persona profiling and naming
- Privacy-safe synthetic behavior generator
- Reproducible CSV outputs and automated tests

## Run locally

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m scripts.run_segmentation
```

The runner writes three files to `outputs/`:

- `segment_assignments.csv`
- `segment_profiles.csv`
- `segmentation_model_comparison.csv`

## Interpretation rule

The selected number of segments is a model result, not a user-entered design choice. Analysts should still review whether the segments are sufficiently distinct, stable, actionable and appropriate for the intended decision. Online experimentation is required before claiming incremental impact.
