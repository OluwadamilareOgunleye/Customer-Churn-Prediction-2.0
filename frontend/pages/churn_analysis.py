"""
Churn Analysis Page - Customer Churn Intelligence Dashboard.
Focuses on operational vulnerability factors: contract structure, internet broadband friction, and payment mechanics.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from frontend.components.header import render_header
from frontend.utils.chart_builder import (
    chart_churn_by_payment_method,
    chart_churn_by_internet_service,
)


def render_page(df: pd.DataFrame, full_df: pd.DataFrame):
    """Render Churn Root Cause & Attrition Hazard Analytics."""
    render_header(total_records=len(df))

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="font-size: 1.25rem; font-weight: 700; color: #0F172A; margin: 0;">
                Root-Cause Churn Analysis & Hazard Vulnerabilities
            </h2>
            <p style="color: #64748B; font-size: 0.875rem; margin: 4px 0 0 0;">
                Investigate billing friction points, broadband infrastructure differences, and subscriber lifecycle hazard stages.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Visuals
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_payment = chart_churn_by_payment_method(df)
        st.plotly_chart(fig_payment, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_internet = chart_churn_by_internet_service(df)
        st.plotly_chart(fig_internet, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)

    # Lifecycle Attrition Hazard Window Deep Dive
    st.markdown('<div class="section-heading">⏳ Lifecycle Hazard Stage Comparison</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    churners = df[df["Churn Value"] == 1]
    total_churners = len(churners)
    y1_churn = (churners["Tenure Months"] <= 12).sum() if total_churners > 0 else 0
    y1_pct = (y1_churn / total_churners * 100.0) if total_churners > 0 else 0

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left: 4px solid #EF4444;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #EF4444; text-transform: uppercase;">
                    Early Hazard Window (0–12 Mos)
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; margin: 6px 0;">
                    {y1_pct:.1f}% of All Churn
                </div>
                <div style="font-size: 0.8rem; color: #64748B;">
                    {y1_churn:,} departing accounts lost during initial customer onboarding.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    echeck_churners = (churners["Payment Method"] == "Electronic check").sum() if total_churners > 0 else 0
    echeck_pct = (echeck_churners / total_churners * 100.0) if total_churners > 0 else 0
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left: 4px solid #F59E0B;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #F59E0B; text-transform: uppercase;">
                    Payment Friction Concentration
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; margin: 6px 0;">
                    {echeck_pct:.1f}% in Electronic Check
                </div>
                <div style="font-size: 0.8rem; color: #64748B;">
                    {echeck_churners:,} churned accounts used unautomated paperless billing checks.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    fiber_churners = (churners["Internet Service"] == "Fiber optic").sum() if total_churners > 0 else 0
    fiber_pct = (fiber_churners / total_churners * 100.0) if total_churners > 0 else 0
    with col3:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left: 4px solid #2563EB;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #2563EB; text-transform: uppercase;">
                    High-ARPU Fiber Attrition
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; margin: 6px 0;">
                    {fiber_pct:.1f}% in Fiber Optic
                </div>
                <div style="font-size: 0.8rem; color: #64748B;">
                    {fiber_churners:,} departures paid premium prices without tech support protection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)

    # Detailed Cross-Tabulation Matrix
    st.markdown('<div class="section-heading">📊 Internet Service & Contract Vulnerability Matrix</div>', unsafe_allow_html=True)
    matrix = df.groupby(["Internet Service", "Contract"], observed=False).agg(
        Total=("Churn Value", "count"),
        Churned=("Churn Value", lambda x: (x == 1).sum()),
        Avg_Monthly_Bill=("Monthly Charges", "mean"),
    ).reset_index()
    matrix["Churn_Rate_%"] = (matrix["Churned"] / matrix["Total"]) * 100.0
    matrix = matrix.sort_values(by="Churn_Rate_%", ascending=False)

    display_mat = matrix.copy()
    display_mat["Total"] = display_mat["Total"].apply(lambda v: f"{v:,}")
    display_mat["Churned"] = display_mat["Churned"].apply(lambda v: f"{v:,}")
    display_mat["Avg_Monthly_Bill"] = display_mat["Avg_Monthly_Bill"].apply(lambda v: f"${v:.2f}")
    display_mat["Churn_Rate_%"] = display_mat["Churn_Rate_%"].apply(lambda v: f"{v:.1f}%")

    display_mat.columns = [
        "Internet Service Type",
        "Contract Duration",
        "Total Subscribers",
        "Churned Subscribers",
        "Avg Monthly Bill",
        "Churn Rate (%)",
    ]
    st.dataframe(display_mat, use_container_width=True, hide_index=True)
