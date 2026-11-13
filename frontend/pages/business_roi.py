"""
Business Insights & ROI Page - Customer Churn Intelligence Dashboard.
Features the Strategic Retention Playbook and an interactive Campaign Financial ROI Simulator.
"""

from pathlib import Path
import json
import streamlit as st

from frontend.components.header import render_header

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
ROI_JSON = PROJECT_ROOT / "reports" / "phase_19" / "roi_simulation_summary.json"


def render_page(df=None, full_df=None):
    """Render Business Playbook & Retention Campaign Simulator."""
    render_header(total_records=7043)

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="font-size: 1.25rem; font-weight: 700; color: #0F172A; margin: 0;">
                Strategic Retention Playbook & Financial ROI Simulator
            </h2>
            <p style="color: #64748B; font-size: 0.875rem; margin: 4px 0 0 0;">
                Translate machine learning predictions into quantified operational interventions and executive ROI business cases.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3 Playbook Initiatives
    st.markdown('<div class="section-heading">📘 Core Strategic Retention Playbook</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="kpi-card" style="border-top: 4px solid #2563EB;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #2563EB; text-transform: uppercase;">
                    Initiative 1
                </div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; margin: 4px 0;">
                    First-Year Shield
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-bottom: 10px;">
                    Targeting months 1–12 subscriber onboarding drop-off.
                </div>
                <div style="font-size: 0.78rem; color: #334155; line-height: 1.4;">
                    • Automated Day-30 check-in call by customer care.<br>
                    • Free 3-month speed boost trial on broadband.<br>
                    • Proactive router diagnostic check.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="kpi-card" style="border-top: 4px solid #10B981;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #10B981; text-transform: uppercase;">
                    Initiative 2
                </div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; margin: 4px 0;">
                    Annual Lock Incentive
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-bottom: 10px;">
                    Converting volatile month-to-month contracts to 12+ months.
                </div>
                <div style="font-size: 0.78rem; color: #334155; line-height: 1.4;">
                    • 10% monthly discount locked for 1-year agreement.<br>
                    • $5/month billing discount for autopay setup.<br>
                    • Free premium streaming service bundle for 6 mos.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="kpi-card" style="border-top: 4px solid #F59E0B;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #F59E0B; text-transform: uppercase;">
                    Initiative 3
                </div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; margin: 4px 0;">
                    Broadband Support Bundle
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-bottom: 10px;">
                    Insulating premium Fiber subscribers against friction.
                </div>
                <div style="font-size: 0.78rem; color: #334155; line-height: 1.4;">
                    • Complimentary Tech Support for accounts paying >$70.<br>
                    • Next-day technician priority guarantee.<br>
                    • Free online security & malware defense pack.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)

    # Financial ROI Simulator
    st.markdown('<div class="section-heading">💰 Interactive Campaign ROI Simulator</div>', unsafe_allow_html=True)

    col_sim_inputs, col_sim_results = st.columns([1.1, 1.4])

    with col_sim_inputs:
        st.markdown("**Campaign Strategy Assumptions**")
        targeted_cohort = st.slider("Targeted Subscriber Cohort Size", min_value=50, max_value=2000, value=531, step=25)
        clv_value = st.slider("Customer Lifetime Value (CLV $ USD)", min_value=400, max_value=3000, value=1200, step=50)
        offer_cost = st.slider("Intervention Offer Cost per Customer ($ USD)", min_value=10, max_value=200, value=50, step=5)
        success_rate = st.slider("Intervention Success / Save Rate (%)", min_value=5, max_value=60, value=25, step=1)
        precision_rate = st.slider("Targeting Precision (% Actual Churners)", min_value=20, max_value=90, value=54, step=1)

    # Reactive Calculations
    true_churners_targeted = targeted_cohort * (precision_rate / 100.0)
    saved_subscribers = int(round(true_churners_targeted * (success_rate / 100.0)))
    total_campaign_cost = float(targeted_cohort * offer_cost)
    gross_revenue_saved = float(saved_subscribers * clv_value)
    net_profit = float(gross_revenue_saved - total_campaign_cost)
    roi_pct = float((net_profit / total_campaign_cost) * 100.0) if total_campaign_cost > 0 else 0.0

    with col_sim_results:
        st.markdown("**Projected Financial Outcome**")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Total Campaign Expenditure</div>
                    <div class="kpi-value" style="color: #64748B;">${total_campaign_cost:,.0f}</div>
                    <div class="kpi-subtext">{targeted_cohort:,} accounts × ${offer_cost}/cust</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Subscribers Rescued</div>
                    <div class="kpi-value" style="color: #10B981;">{saved_subscribers:,}</div>
                    <div class="kpi-subtext">Prevented customer departures</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="margin-top: 12px;"></div>', unsafe_allow_html=True)

        r3, r4 = st.columns(2)
        with r3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Gross Revenue Saved</div>
                    <div class="kpi-value" style="color: #2563EB;">${gross_revenue_saved:,.0f}</div>
                    <div class="kpi-subtext">Preserved account lifetime value</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r4:
            profit_color = "#10B981" if net_profit >= 0 else "#EF4444"
            st.markdown(
                f"""
                <div class="kpi-card" style="border: 2px solid {profit_color};">
                    <div class="kpi-title">Net Financial Profit & ROI</div>
                    <div class="kpi-value" style="color: {profit_color};">${net_profit:,.0f}</div>
                    <div class="kpi-subtext" style="color: {profit_color}; font-weight: 700;">
                        <b>{roi_pct:+.1f}% Net Return on Spend</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)

    # Strategy comparison image from reports
    st.markdown('<div class="section-heading">📊 Validated Test-Set Benchmark ROI Comparison</div>', unsafe_allow_html=True)
    roi_img = FIGURES_DIR / "22_business_retention_roi_comparison.png"
    if roi_img.exists():
        st.image(str(roi_img), use_container_width=True, caption="Figure 22: Retention Campaign Net Profit & ROI Across Decision Thresholds")
