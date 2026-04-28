"""
Simple Streamlit demo app for personalization recommendations.

Run:
streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Personalization Engine", layout="wide")

st.title("End-to-End Personalization Engine")
st.write(
    "This demo illustrates how a recommendation system can personalize content "
    "based on user behavior and content metadata."
)

user_segment = st.selectbox(
    "Select user segment",
    ["High-intent readers", "Exploratory browsers", "Deal seekers", "Loyal subscribers", "At-risk users"]
)

st.subheader("Recommended Content")
demo_recs = pd.DataFrame({
    "Rank": [1, 2, 3, 4, 5],
    "Recommendation": [
        "High-value product comparison guide",
        "Expert buying advice article",
        "Personalized newsletter module",
        "Trending topic explainer",
        "Conversion-focused offer page"
    ],
    "Predicted Engagement Score": [0.89, 0.84, 0.80, 0.76, 0.72],
    "Business Rationale": [
        "Strong alignment with prior high-intent behavior",
        "Matches observed content category preference",
        "Likely to increase repeat engagement",
        "Captures emerging interest signal",
        "Supports downstream conversion"
    ]
})

st.dataframe(demo_recs, use_container_width=True)

st.subheader("Executive Recommendation")
st.write(
    f"For the selected segment, prioritize content that combines demonstrated intent, "
    f"category affinity, and conversion proximity. Use experimentation to validate "
    f"incremental lift before scaling."
)
