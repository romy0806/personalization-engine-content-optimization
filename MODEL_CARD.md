# MIND Behavioral Segmentation Model Card

## Intended use

Discover explainable reader groups from Microsoft MIND interaction logs for content-personalization analysis and experiment design. Segment assignments are decision-support signals, not causal effects or permanent user identities.

## Data and inputs

The model consumes the published MIND `behaviors.tsv` and `news.tsv` schemas. User features summarize observed reading history, sessions, recommendation exposure, clicks, recency, and topic diversity across the supplied observation window.

## Methodology

1. Parse and validate MIND behavior and news records.
2. Aggregate MIND interactions to one row per user.
3. Impute numeric values, cap outliers, and apply robust scaling.
4. Compare K-Means and Gaussian Mixture candidates across 2–8 segments.
5. Evaluate separation, compactness, bootstrap stability, and cluster-size viability.
6. Select a model using a documented composite score calculated on MIND train only.
7. Name segments from their behavior relative to the MIND training population.
8. Freeze preprocessing, model parameters, cluster count, and persona names.
9. Score MIND dev without refitting and quantify segment and feature drift with PSI.

## Limitations

- MIND represents news recommendation behavior, not purchases, subscriptions, or long-term customer value.
- Recency is relative to the latest timestamp in the supplied MIND split.
- Offline clusters and recommendation metrics do not prove incremental business impact.
- Persona labels are descriptive summaries and require analyst review.
- Segment structure may change across MIND train/dev splits or new observation windows.

## Monitoring and validation

Track feature drift, cluster shares, bootstrap Adjusted Rand Index, assignment strength, and
downstream recommendation quality. Model selection and bootstrap stability are calculated on
MIND train. MIND dev is an untouched validation population used for separation, assignment
strength, cluster-size viability, segment-distribution PSI, and feature PSI. Controlled
experiments are still required before claiming engagement lift.
