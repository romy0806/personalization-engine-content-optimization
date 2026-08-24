"""Meeting-ready Streamlit demo for validated MIND user segmentation.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "demo_data"

st.set_page_config(
    page_title="Personalization Segmentation Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_demo_results() -> dict[str, pd.DataFrame]:
    """Load compact, versioned outputs produced from real MIND train/dev data."""
    return {
        "models": pd.read_csv(DATA_DIR / "model_comparison.csv"),
        "segments": pd.read_csv(DATA_DIR / "segment_comparison.csv"),
        "profiles": pd.read_csv(DATA_DIR / "validation_profiles.csv"),
        "metrics": pd.read_csv(DATA_DIR / "validation_metrics.csv"),
        "drift": pd.read_csv(DATA_DIR / "feature_drift.csv"),
    }


def metric_value(metrics: pd.DataFrame, name: str) -> float:
    """Return one named validation metric."""
    return float(metrics.loc[metrics["metric"] == name, "value"].iloc[0])


def kpi_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .stApp { background: #f7f5f2; color: #24182d; }
    .block-container { max-width: 1480px; padding-top: 2rem; }
    section[data-testid="stSidebar"] { background: #261737; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
    color: #f7effa !important;
    }
    .hero { padding: 2rem 2.2rem; border-radius: 22px; color: white;
      background: linear-gradient(125deg, #241433 0%, #6f3f7c 58%, #ed7655 100%);
      box-shadow: 0 18px 42px rgba(40, 24, 48, .18); margin-bottom: 1.2rem; }
    .hero h1 { color: white; font-size: 2.55rem; margin: .35rem 0 .55rem; }
    .hero p { color: #f9edf7; font-size: 1.05rem; margin: 0; max-width: 900px; }
    .eyebrow { color: #ffd5c6; font-size: .76rem; font-weight: 800;
      letter-spacing: .16em; text-transform: uppercase; }
    .pill-row { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.15rem; }
    .pill { border:1px solid rgba(255,255,255,.3); border-radius:999px;
      padding:.38rem .7rem; font-size:.8rem; font-weight:700; }
    .kpi-card { background:white; border:1px solid #e3dbe6; border-radius:16px;
      padding:1rem 1.05rem; min-height:118px; box-shadow:0 5px 16px rgba(36,24,45,.06); }
    .kpi-label { color:#746a78; font-size:.69rem; font-weight:800;
      letter-spacing:.09em; text-transform:uppercase; }
    .kpi-value { color:#281a31; font-size:1.65rem; font-weight:800; margin:.35rem 0 .2rem; }
    .kpi-help { color:#746a78; font-size:.78rem; line-height:1.35; }
    .callout { padding:1rem 1.1rem; border-left:5px solid #ed7655;
      background:#fff0e9; border-radius:0 12px 12px 0; color:#66392f; }
    .method-step { background:white; border:1px solid #e3dbe6; border-radius:14px;
      padding:1rem; min-height:145px; }
    .method-number { color:#ed7655; font-weight:900; font-size:.75rem; }
    .method-title { color:#281a31; font-weight:800; margin:.35rem 0; }
    .method-copy { color:#746a78; font-size:.84rem; line-height:1.45; }
    div[data-testid="stDataFrame"] { border:1px solid #e3dbe6; border-radius:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

data = load_demo_results()
models = data["models"]
segments = data["segments"]
profiles = data["profiles"]
metrics = data["metrics"]
drift = data["drift"]

st.sidebar.title("Demo controls")
st.sidebar.caption("Explore the frozen model and its held-out validation results.")
selected_segment = st.sidebar.selectbox(
    "Persona",
    profiles.sort_values("validation_share")["segment_name"].tolist(),
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Source**")
st.sidebar.caption("Microsoft MIND train and dev behavior/news files")
st.sidebar.markdown("**Training population**")
st.sidebar.caption("50,000 sampled train users")
st.sidebar.markdown("**Validation population**")
st.sidebar.caption("50,000 held-out dev users")
st.sidebar.info(
    "The app reads compact, versioned result extracts so the meeting demo is fast. "
    "The full pipeline remains reproducible from the repository scripts."
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Validated personalization segmentation</div>
      <h1>MIND Audience Intelligence Lab</h1>
      <p>Discover behavior-based audience personas, inspect why the model selected five
      clusters, and verify that those personas remain usable on held-out users.</p>
      <div class="pill-row">
        <span class="pill">Real MIND data</span>
        <span class="pill">Train → freeze → dev validation</span>
        <span class="pill">K-Means + GMM evaluated</span>
        <span class="pill">No subscription or purchase claims</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_columns = st.columns(5)
with kpi_columns[0]:
    kpi_card("Selected model", "K-Means", "Chosen using quality, stability, and size guardrails")
with kpi_columns[1]:
    kpi_card("Personas", "5", "Automatically selected from candidates k=2 through k=8")
with kpi_columns[2]:
    kpi_card(
        "Train stability", f"{models.iloc[0]['stability_ari']:.3f}", "Bootstrap adjusted Rand index"
    )
with kpi_columns[3]:
    kpi_card(
        "Held-out silhouette",
        f"{metric_value(metrics, 'silhouette'):.3f}",
        "Cluster separation on MIND dev",
    )
with kpi_columns[4]:
    kpi_card(
        "Segment PSI",
        f"{metric_value(metrics, 'segment_distribution_psi'):.3f}",
        "Low aggregate distribution shift",
    )

overview_tab, explorer_tab, methodology_tab, validation_tab = st.tabs(
    ["Executive overview", "Persona explorer", "Methodology", "Validation & caveats"]
)

with overview_tab:
    st.subheader("What the model discovered")
    st.caption("Audience share by persona in train and held-out dev populations.")
    share_chart = segments.set_index("segment_name")[["train_share", "validation_share"]]
    share_chart.columns = ["Train", "Held-out dev"]
    st.bar_chart(share_chart, y_label="User share", color=["#75489a", "#ed7655"])

    left, right = st.columns([1.2, 1])
    with left:
        display_segments = segments[
            [
                "segment_name",
                "train_users",
                "train_share",
                "validation_users",
                "validation_share",
                "share_delta",
            ]
        ].copy()
        display_segments.columns = [
            "Persona",
            "Train users",
            "Train share",
            "Dev users",
            "Dev share",
            "Share delta",
        ]
        st.dataframe(
            display_segments.style.format(
                {"Train share": "{:.1%}", "Dev share": "{:.1%}", "Share delta": "{:+.1%}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.markdown(
            """
            <div class="callout"><b>Decision use:</b> these personas can guide audience
            strategy, content packaging, experiment design, and recommendation-policy
            hypotheses. They do not themselves prove incremental lift.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### What is production-worthy")
        st.markdown(
            "- Frozen preprocessing and cluster model\n"
            "- Deterministic persona naming\n"
            "- Held-out validation and drift monitoring\n"
            "- Versioned extracts for reproducible review"
        )

with explorer_tab:
    profile = profiles.loc[profiles["segment_name"] == selected_segment].iloc[0]
    comparison = segments.loc[segments["segment_name"] == selected_segment].iloc[0]
    st.subheader(selected_segment)
    st.caption(f"Defined most strongly by: {profile['defining_features'].replace(',', ', ')}")

    persona_columns = st.columns(4)
    with persona_columns[0]:
        kpi_card(
            "Held-out users",
            f"{int(profile['validation_users']):,}",
            f"{profile['validation_share']:.1%} of dev users",
        )
    with persona_columns[1]:
        kpi_card("History length", f"{profile['history_length']:.1f}", "Average prior articles")
    with persona_columns[2]:
        kpi_card(
            "CTR", f"{profile['click_through_rate']:.1%}", "Clicks divided by shown impressions"
        )
    with persona_columns[3]:
        kpi_card("Clicks/session", f"{profile['clicks_per_session']:.2f}", "Engagement intensity")

    feature_values = pd.DataFrame(
        {
            "Feature": [
                "Avg slate size",
                "Category diversity",
                "Subcategory diversity",
                "Dominant category share",
            ],
            "Value": [
                profile["avg_impression_slate_size"],
                profile["category_diversity"],
                profile["subcategory_diversity"],
                profile["dominant_category_share"],
            ],
        }
    )
    st.dataframe(feature_values, hide_index=True, use_container_width=True)

    activation_copy = {
        "High-intent clickers": "Reduce low-relevance exposure and test stronger category-affinity ranking on larger content slates.",
        "Highly responsive readers": "Test timely, high-confidence recommendations and deeper follow-on journeys after a click.",
        "Deep-history loyalists": "Use long-term preference signals and test discovery modules that broaden an already deep relationship.",
        "Cross-topic explorers": "Favor diverse recommendation sets and test novelty-aware ranking across adjacent topics.",
        "Topic specialists": "Prioritize concentrated topic depth, then test carefully bounded adjacent-topic exploration.",
    }
    st.markdown("#### Activation hypothesis")
    st.info(activation_copy[selected_segment])
    st.caption(
        f"Train-to-dev share changed by {comparison['share_delta']:+.1%}. "
        "Treat this statement as a hypothesis requiring an online experiment."
    )

with methodology_tab:
    st.subheader("Methodology: what happens and why")
    method_columns = st.columns(5)
    steps = [
        (
            "01",
            "Source",
            "Parse real MIND behaviors.tsv and news.tsv; aggregate records to one row per user.",
        ),
        (
            "02",
            "Features",
            "Create history depth, CTR, clicks/session, slate size, topic diversity, and concentration features.",
        ),
        (
            "03",
            "Prepare",
            "Median-impute, cap extremes, and robust-scale using parameters learned only from train.",
        ),
        (
            "04",
            "Select",
            "Compare K-Means and GMM for k=2–8 using separation, bootstrap stability, and size guardrails.",
        ),
        (
            "05",
            "Validate",
            "Freeze the winning train pipeline, assign dev users without refitting, then measure quality and drift.",
        ),
    ]
    for column, (number, title, copy) in zip(method_columns, steps, strict=True):
        with column:
            st.markdown(
                f"<div class='method-step'><div class='method-number'>{number}</div>"
                f"<div class='method-title'>{title}</div><div class='method-copy'>{copy}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Why five clusters?")
    model_display = models.copy()
    model_display["candidate"] = (
        model_display["algorithm"].str.replace("_", " ").str.title()
        + " · k="
        + model_display["clusters"].astype(str)
    )
    model_display["plot_size"] = model_display["selection_score"].clip(lower=0.05)
    st.scatter_chart(
        model_display,
        x="silhouette",
        y="stability_ari",
        color="business_viable",
        size="plot_size",
    )
    st.dataframe(
        model_display[
            [
                "candidate",
                "silhouette",
                "davies_bouldin",
                "stability_ari",
                "smallest_cluster_share",
                "largest_cluster_share",
                "business_viable",
                "selection_score",
            ]
        ].style.format(
            {
                "silhouette": "{:.3f}",
                "davies_bouldin": "{:.3f}",
                "stability_ari": "{:.3f}",
                "smallest_cluster_share": "{:.1%}",
                "largest_cluster_share": "{:.1%}",
                "selection_score": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "The highest raw silhouette was not selected because its largest cluster exceeded the business-size guardrail. "
        "The chosen solution balances separation, repeatability, and actionable segment sizes."
    )

with validation_tab:
    st.subheader("Held-out validation")
    st.caption("MIND dev is never used to choose or refit the clusters.")
    validation_columns = st.columns(4)
    with validation_columns[0]:
        kpi_card(
            "Dev users",
            f"{int(metric_value(metrics, 'validation_users')):,}",
            "Assigned with frozen train model",
        )
    with validation_columns[1]:
        kpi_card(
            "Davies–Bouldin",
            f"{metric_value(metrics, 'davies_bouldin'):.3f}",
            "Lower indicates better separation",
        )
    with validation_columns[2]:
        kpi_card(
            "Mean strength",
            f"{metric_value(metrics, 'mean_assignment_strength'):.3f}",
            "Relative distance diagnostic",
        )
    with validation_columns[3]:
        kpi_card(
            "P10 strength",
            f"{metric_value(metrics, 'p10_assignment_strength'):.3f}",
            "Lower-tail assignment diagnostic",
        )

    st.markdown("#### Feature drift (PSI)")
    st.bar_chart(drift.set_index("feature")["psi"], y_label="PSI", color="#75489a")
    st.dataframe(
        drift[["feature", "psi", "scaled_mean_delta"]].style.format(
            {"psi": "{:.3f}", "scaled_mean_delta": "{:+.3f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("#### Important limits")
    st.warning(
        "MIND records news exposure and clicks. It does not contain subscriptions, purchases, revenue, or causal treatment effects. "
        "Persona labels are behavioral summaries; assignment strength is not a calibrated probability; activation ideas require A/B testing."
    )
