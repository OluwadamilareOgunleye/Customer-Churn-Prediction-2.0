"""
Customer Analytics Page - Customer Churn Intelligence Dashboard.
Deep exploration of subscriber demographics, household structure, service adoption, and billing distribution.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from frontend.components.header import render_header
from frontend.utils.chart_builder import (
    chart_churn_by_living_situation,
    chart_churn_by_services_count,
    chart_charges_by_churn_status,
    chart_tenure_distribution,
)


def render_page(df: pd.DataFrame, full_df: pd.DataFrame):
    """Render Demographics and Service Adoption Analytics."""
    render_header(total_records=len(df))

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="font-size: 1.25rem; font-weight: 700; color: #0F172A; margin: 0;">
                Subscriber Demographics & Service Adoption Analytics
            </h2>
            <p style="color: #64748B; font-size: 0.875rem; margin: 4px 0 0 0;">
                Analyze household composition, optional add-on product depth, and billing distribution across accounts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Chart Grid 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_living = chart_churn_by_living_situation(df)
        st.plotly_chart(fig_living, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_services = chart_churn_by_services_count(df)
        st.plotly_chart(fig_services, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

    # Chart Grid 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_charges = chart_charges_by_churn_status(df)
        st.plotly_chart(fig_charges, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_tenure = chart_tenure_distribution(df)
        st.plotly_chart(fig_tenure, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)

    # Tabular Breakdown Section
    st.markdown('<div class="section-heading">📋 Cohort Breakdown & Empirical Statistics</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Household Composition Breakdown", "Add-on Services Depth Breakdown"])

    with tab1:
        summary_living = df.groupby("Living Situation", observed=False).agg(
            Total_Accounts=("Churn Value", "count"),
            Churned_Accounts=("Churn Value", lambda x: (x == 1).sum()),
            Avg_Monthly_Bill=("Monthly Charges", "mean"),
            Total_MRR=("Monthly Charges", "sum"),
        ).reset_index()
        summary_living["Churn_Rate_%"] = (summary_living["Churned_Accounts"] / summary_living["Total_Accounts"]) * 100.0
        summary_living = summary_living.sort_values(by="Churn_Rate_%", ascending=False)

        # Format display
        display_living = summary_living.copy()
        display_living["Total_Accounts"] = display_living["Total_Accounts"].apply(lambda v: f"{v:,}")
        display_living["Churned_Accounts"] = display_living["Churned_Accounts"].apply(lambda v: f"{v:,}")
        display_living["Avg_Monthly_Bill"] = display_living["Avg_Monthly_Bill"].apply(lambda v: f"${v:.2f}")
        display_living["Total_MRR"] = display_living["Total_MRR"].apply(lambda v: f"${v:,.2f}")
        display_living["Churn_Rate_%"] = display_living["Churn_Rate_%"].apply(lambda v: f"{v:.1f}%")

        display_living.columns = [
            "Household Composition",
            "Total Accounts",
            "Churned Accounts",
            "Avg Monthly Bill",
            "Total MRR ($)",
            "Churn Rate (%)",
        ]
        st.dataframe(display_living, use_container_width=True, hide_index=True)

    with tab2:
        summary_services = df.groupby("Service Count", observed=False).agg(
            Total_Accounts=("Churn Value", "count"),
            Churned_Accounts=("Churn Value", lambda x: (x == 1).sum()),
            Avg_Monthly_Bill=("Monthly Charges", "mean"),
            Total_MRR=("Monthly Charges", "sum"),
        ).reset_index()
        summary_services["Churn_Rate_%"] = (summary_services["Churned_Accounts"] / summary_services["Total_Accounts"]) * 100.0
        summary_services = summary_services.sort_values(by="Service Count", ascending=True)

        display_services = summary_services.copy()
        display_services["Total_Accounts"] = display_services["Total_Accounts"].apply(lambda v: f"{v:,}")
        display_services["Churned_Accounts"] = display_services["Churned_Accounts"].apply(lambda v: f"{v:,}")
        display_services["Avg_Monthly_Bill"] = display_services["Avg_Monthly_Bill"].apply(lambda v: f"${v:.2f}")
        display_services["Total_MRR"] = display_services["Total_MRR"].apply(lambda v: f"${v:,.2f}")
        display_services["Churn_Rate_%"] = display_services["Churn_Rate_%"].apply(lambda v: f"{v:.1f}%")

        display_services.columns = [
            "Add-on Services Count (0–6)",
            "Total Accounts",
            "Churned Accounts",
            "Avg Monthly Bill",
            "Total MRR ($)",
            "Churn Rate (%)",
        ]
        st.dataframe(display_services, use_container_width=True, hide_index=True)
