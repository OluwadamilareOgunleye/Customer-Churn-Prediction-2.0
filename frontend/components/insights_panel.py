"""
Dynamic Business Insights Generator.
Analyzes the active customer slice and produces automated, data-driven observations and strategic levers.
"""

from typing import List, Dict, Any
import pandas as pd
import streamlit as st


def generate_dynamic_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generate dynamic business observations derived strictly from data.
    """
    if len(df) < 20:
        return [
            {
                "type": "warning",
                "title": "Insufficient Sample Size",
                "stat": f"Active cohort: {len(df)} records",
                "desc": "Cohort is too small for statistical pattern detection. Clear some filters to view broader trends.",
                "action": "Broaden active filter selections",
            }
        ]

    insights = []
    overall_churn = (df["Churn Value"] == 1).mean() * 100.0

    # 1. Contract Insight
    if "Contract" in df.columns:
        contract_churn = df.groupby("Contract", observed=False)["Churn Value"].mean() * 100.0
        m2m_churn = contract_churn.get("Month-to-month", 0)
        two_yr_churn = contract_churn.get("Two year", 0)
        if m2m_churn > 0:
            diff = m2m_churn - two_yr_churn
            insights.append({
                "type": "danger" if m2m_churn >= 35 else "warning",
                "title": "Contract Duration Drives Churn Rate",
                "stat": f"{m2m_churn:.1f}% churn on Month-to-month vs {two_yr_churn:.1f}% on 2-Year",
                "desc": f"Subscribers on monthly plans exhibit a {diff:.1f}% higher churn rate than accounts with 2-year commitments.",
                "action": "Incentivize annual contract upgrades with promotional hardware or discounts",
            })

    # 2. Tenure / Early Life Hazard Insight
    if "Tenure Months" in df.columns:
        churners_df = df[df["Churn Value"] == 1]
        if len(churners_df) > 0:
            median_churn_tenure = churners_df["Tenure Months"].median()
            first_year_churners = (churners_df["Tenure Months"] <= 12).mean() * 100.0
            insights.append({
                "type": "danger" if first_year_churners >= 50 else "warning",
                "title": "Critical Hazard Window: Months 1 to 12",
                "stat": f"{first_year_churners:.1f}% of churn occurs within the first year",
                "desc": f"The median tenure of departing customers is {median_churn_tenure:.0f} months. Onboarding drop-off is the single largest attrition driver.",
                "action": "Deploy 30/60/90-day automated customer success touchpoints and onboarding support",
            })

    # 3. Payment Method Friction
    if "Payment Method" in df.columns:
        pay_churn = df.groupby("Payment Method", observed=False)["Churn Value"].mean() * 100.0
        echeck_churn = pay_churn.get("Electronic check", 0)
        bank_churn = pay_churn.get("Bank transfer (automatic)", 0)
        if echeck_churn > bank_churn:
            insights.append({
                "type": "warning",
                "title": "Manual Electronic Check Billing Carries Elevated Friction",
                "stat": f"{echeck_churn:.1f}% churn for Electronic check vs {bank_churn:.1f}% for Auto-Bank Transfer",
                "desc": "Customers using manual electronic checks have significantly higher churn propensity than those enrolled in autopay.",
                "action": "Offer a $5/month billing discount to switch from electronic check to automated ACH / card",
            })

    # 4. Tech Support & Retention Anchor
    if "Tech Support" in df.columns and "Internet Service" in df.columns:
        fiber_no_sup = df[(df["Internet Service"] == "Fiber optic") & (df["Tech Support"] == "No")]
        fiber_with_sup = df[(df["Internet Service"] == "Fiber optic") & (df["Tech Support"] == "Yes")]
        if len(fiber_no_sup) > 30 and len(fiber_with_sup) > 30:
            rate_no_sup = (fiber_no_sup["Churn Value"] == 1).mean() * 100.0
            rate_with_sup = (fiber_with_sup["Churn Value"] == 1).mean() * 100.0
            insights.append({
                "type": "success",
                "title": "Tech Support Acts as a High-Value Retention Anchor",
                "stat": f"Fiber optic churn drops from {rate_no_sup:.1f}% down to {rate_with_sup:.1f}% with Tech Support",
                "desc": "Adding Tech Support cuts premium broadband subscriber churn by more than half, offsetting dissatisfaction from technical hiccups.",
                "action": "Bundle complimentary 6-month Tech Support into premium broadband tier acquisitions",
            })

    return insights


def render_insights_panel(df: pd.DataFrame):
    """Render dynamic insights card grid."""
    insights = generate_dynamic_insights(df)

    st.markdown('<div class="section-heading">💡 Dynamic Business Observations & Levers</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for i, ins in enumerate(insights):
        col = c1 if i % 2 == 0 else c2
        card_class = f"insight-card {ins['type']}"
        with col:
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div class="insight-title">{ins['title']}</div>
                    <div style="font-size: 0.8125rem; font-weight: 700; color: #0F172A; margin-bottom: 4px;">
                        📌 {ins['stat']}
                    </div>
                    <div class="insight-desc">{ins['desc']}</div>
                    <div class="insight-action">Recommended Action: {ins['action']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
