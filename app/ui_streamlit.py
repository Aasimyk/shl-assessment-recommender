# app/ui_streamlit.py
from __future__ import annotations

import io
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import streamlit as st  # [web:54]

# Ensure project root is on sys.path when run via "streamlit run app/ui_streamlit.py"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.main import _search_catalog  # noqa: E402


def _to_dataframe(results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Rank": r["rank"],
                "Assessment": r["name"],
                "URL": r["url"],
                "Duration (mins)": r["duration_minutes"],
                "Remote testing": r["remote_testing"],
                "Adaptive / IRT": r["adaptive_irt"],
                "Test Types": r["test_type_codes"],
                "Job Levels": r["job_levels"],
                "Languages": r["languages"],
                "Description": r["description"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="SHL Assessment Recommender",
        layout="wide",
    )

    st.markdown(
        """
        <h1 style="text-align:center; margin-bottom:0.2rem;">
        SHL Assessment Recommender
        </h1>
        <p style="text-align:center; font-size:0.95rem; color:#666;">
        Describe the role, skills, and constraints to get suitable SHL assessments.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        col_left, col_right = st.columns([2, 3], gap="large")

        with col_left:
            st.markdown("#### Query")
            default_example = (
                "Hiring a mid‑level Java developer for a product company. "
                "Strong problem solving, coding, debugging and communication. "
                "Total assessment time should be under 90 minutes."
            )
            query = st.text_area(
                label="Job description or hiring query",
                label_visibility="collapsed",
                value=default_example,
                height=200,
                placeholder="Describe the role, skills, and time budget for the assessment...",
            )

            top_k = st.slider("Number of recommendations", 3, 20, 10)
            run_btn = st.button("Search", type="primary", use_container_width=True)

        with col_right:
            st.markdown("#### Recommendations")
            results_placeholder = st.empty()

    if run_btn and query.strip():
        with st.spinner("Finding best‑fit assessments..."):
            results = _search_catalog(query.strip(), top_k=top_k)

        if not results:
            results_placeholder.info("No assessments found for this query.")
            return

        df = _to_dataframe(results)
        results_placeholder.dataframe(df, use_container_width=True)

        # CSV download
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        st.download_button(
            label="Download results as CSV",
            data=csv_bytes,
            file_name="shl_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        results_placeholder = st.empty()
        results_placeholder.info("Enter a query and click Search to see recommendations.")


if __name__ == "__main__":
    main()
