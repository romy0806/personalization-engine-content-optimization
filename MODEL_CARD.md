# Behavioral Segmentation Model Card

## Intended use

Discover explainable behavioral audience groups for content-personalization analysis and experiment design. Segment assignments are decision-support signals, not proof that a treatment causes engagement or conversion.

## Inputs

One row per user, calculated over a documented observation window. Required features include recency, sessions, content consumption, click-through rate, category diversity, repeat visits, session depth, and high-intent actions.

## Methodology

1. Validate schema and minimum population size.
2. Impute missing numeric values using fitted medians.
3. Cap extreme observations using fitted 1st and 99th percentiles.
4. Apply robust scaling.
5. Compare K-Means and Gaussian Mixture candidates across 2–8 segments.
6. Evaluate separation, compactness, stability and cluster-size viability.
7. Select a deployable model using a documented composite score.
8. Create persona names from cluster behavior relative to the total population.

## Selection evidence

- Silhouette score: higher is better.
- Davies–Bouldin index: lower is better.
- Calinski–Harabasz score: higher is better.
- Bootstrap Adjusted Rand Index: higher is more stable.
- Cluster shares: reject solutions with extremely small or dominant segments.

## Limitations

- Personas describe observed behavior and should not be interpreted as causal identities.
- Segment meaning can drift when content, acquisition mix, seasonality or product design changes.
- Sparse or newly acquired users may have low-confidence assignments.
- Sensitive or protected attributes must not be used for targeting without an approved policy and fairness review.
- Online experiments are required before claiming incremental business impact.

## Monitoring recommendations

Track population stability, feature drift, cluster share changes, membership confidence, segment-level recommendation quality and experiment lift. Refit only after defined drift or performance thresholds are crossed.
