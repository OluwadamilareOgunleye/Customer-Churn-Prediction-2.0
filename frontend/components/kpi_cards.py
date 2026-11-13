"""
Executive SaaS KPI Card Component.
Renders responsive metrics cards with icons, currency, and baseline comparisons.
"""

from typing import Dict, Any
import streamlit as st


def render_executive_kpis(kpis: Dict[str, Any]):
    """Render 2 rows of 4 enterprise metric cards."""
    # Row 1: Subscriber Volumes & Rates
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Total Customers</span>
                    <span class="kpi-icon">👥</span>
                </div>
                <div class="kpi-value">{kpis['total_customers']:,}</div>
                <div class="kpi-subtext">Active cohort sample size</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Churned Accounts</span>
                    <span class="kpi-icon" style="background: #FEF2F2; color: #EF4444;">📉</span>
                </div>
                <div class="kpi-value" style="color: #EF4444;">{kpis['churned_customers']:,}</div>
                <div class="kpi-subtext">Terminated subscriptions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Retained Accounts</span>
                    <span class="kpi-icon" style="background: #ECFDF5; color: #10B981;">🛡️</span>
                </div>
                <div class="kpi-value" style="color: #10B981;">{kpis['retained_customers']:,}</div>
                <div class="kpi-subtext">Active continuous subscribers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        delta_str = ""
        if kpis.get("churn_rate_delta") is not None:
            sign = "+" if kpis["churn_rate_delta"] > 0 else ""
            color = "#EF4444" if kpis["churn_rate_delta"] > 0 else "#10B981"
            delta_str = f'<span style="color: {color}; font-weight: 600;">{sign}{kpis["churn_rate_delta"]}%</span> vs baseline'
        else:
            delta_str = "Global population average"

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Overall Churn Rate</span>
                    <span class="kpi-icon" style="background: #FEF3C7; color: #F59E0B;">⚡</span>
                </div>
                <div class="kpi-value">{kpis['churn_rate']:.1f}%</div>
                <div class="kpi-subtext">{delta_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Row 2: Financial Metrics & Engagement
    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Monthly Recurring Rev (MRR)</span>
                    <span class="kpi-icon">💰</span>
                </div>
                <div class="kpi-value">${kpis['mrr_total']:,.0f}</div>
                <div class="kpi-subtext">Monthly billed subscriptions (USD)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Historical MRR Lost</span>
                    <span class="kpi-icon" style="background: #FEF2F2; color: #DC2626;">💸</span>
                </div>
                <div class="kpi-value" style="color: #DC2626;">${kpis['revenue_lost']:,.0f}</div>
                <div class="kpi-subtext">Monthly revenue lost from churners</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Avg Monthly Charges</span>
                    <span class="kpi-icon">📊</span>
                </div>
                <div class="kpi-value">${kpis['avg_monthly_charges']:.2f}</div>
                <div class="kpi-subtext">Mean monthly subscriber bill</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Avg Customer Tenure</span>
                    <span class="kpi-icon">⏳</span>
                </div>
                <div class="kpi-value">{kpis['avg_tenure']:.1f} <span style="font-size: 1.1rem; font-weight: 500; color: #64748B;">mos</span></div>
                <div class="kpi-subtext">Relationship duration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
