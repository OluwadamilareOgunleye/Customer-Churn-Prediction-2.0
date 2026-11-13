"""
Customer Segmentation Page - Customer Churn Intelligence Dashboard.
Displays K-Means behavioral clustering personas, profile distributions, and targeted retention strategies.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from frontend.components.header import render_header

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
PROFILES_CSV = PROJECT_ROOT / "reports" / "phase_19" / "customer_persona_profiles.csv"


def render_page(df: pd.DataFrame, full_df: pd.DataFrame):
    """Render Customer Behavioral Personas & Clustering."""
    render_header(total_records=len(df))

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="font-size: 1.25rem; font-weight: 700; color: #0F172A; margin: 0;">
                Behavioral Customer Personas & Unsupervised Clustering
            </h2>
            <p style="color: #64748B; font-size: 0.875rem; margin: 4px 0 0 0;">
                K-Means clustering on behavioral and financial dimensions identifies three distinct subscriber cohorts with distinct retention levers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3 Persona Cards
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="kpi-card" style="border-top: 4px solid #10B981;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #10B981; text-transform: uppercase;">
                    Cluster 1 • 17.3% Base
                </div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 4px 0;">
                    Long-Term Value Loyalists
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-bottom: 8px;">
                    Mature accounts with high tenure and modest bills. High stickiness, low maintenance.
                </div>
                <div style="background: #F8FAFC; border-radius: 6px; padding: 8px; font-size: 0.78rem; color: #334155;">
                    <div>• <b>Avg Tenure:</b> 53.7 Months</div>
                    <div>• <b>Avg Monthly Bill:</b> $34.07</div>
                    <div>• <b>Empirical Churn:</b> 4.5%</div>
                    <div>• <b>High Risk Share:</b> 2.0%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="kpi-card" style="border-top: 4px solid #F59E0B;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #F59E0B; text-transform: uppercase;">
                    Cluster 2 • 23.5% Base
                </div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 4px 0;">
                    New Budget Customers
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-bottom: 8px;">
                    Early-stage subscribers with low bills testing service reliability. Elevated onboarding drop-off.
                </div>
                <div style="background: #F8FAFC; border-radius: 6px; padding: 8px; font-size: 0.78rem; color: #334155;">
                    <div>• <b>Avg Tenure:</b> 9.5 Months</div>
                    <div>• <b>Avg Monthly Bill:</b> $31.87</div>
                    <div>• <b>Empirical Churn:</b> 27.2%</div>
                    <div>• <b>High Risk Share:</b> 39.9%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="kpi-card" style="border-top: 4px solid #EF4444;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #EF4444; text-transform: uppercase;">
                    Cluster 3 • 59.2% Base
                </div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 4px 0;">
                    Tenured Premium Accounts
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-bottom: 8px;">
                    Highest revenue cohort with multi-service bundles, but substantial price sensitivity.
                </div>
                <div style="background: #F8FAFC; border-radius: 6px; padding: 8px; font-size: 0.78rem; color: #334155;">
                    <div>• <b>Avg Tenure:</b> 34.5 Months</div>
                    <div>• <b>Avg Monthly Bill:</b> $85.66</div>
                    <div>• <b>Empirical Churn:</b> 32.7%</div>
                    <div>• <b>High Risk Share:</b> 47.2%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)

    # Cluster Visual & Empirical Profiles Table
    col_img, col_tbl = st.columns([1.1, 0.9])

    with col_img:
        st.markdown('<div class="section-heading">📈 Cluster Distribution Visualization</div>', unsafe_allow_html=True)
        cluster_fig = FIGURES_DIR / "21_customer_segmentation_clusters.png"
        if cluster_fig.exists():
            st.image(str(cluster_fig), use_container_width=True, caption="Figure 21: K-Means Persona Clusters (Tenure vs Monthly Charges)")
        else:
            st.info("Cluster visualization figure pending generation.")

    with col_tbl:
        st.markdown('<div class="section-heading">📋 Persona Statistical Profile</div>', unsafe_allow_html=True)
        if PROFILES_CSV.exists():
            prof_df = pd.read_csv(PROFILES_CSV)
            disp_prof = prof_df.copy()
            disp_prof["customer_count"] = disp_prof["customer_count"].apply(lambda v: f"{v:,}")
            disp_prof["avg_tenure"] = disp_prof["avg_tenure"].apply(lambda v: f"{v:.1f} mos")
            disp_prof["avg_monthly_charges"] = disp_prof["avg_monthly_charges"].apply(lambda v: f"${v:.2f}")
            disp_prof["actual_churn_rate"] = disp_prof["actual_churn_rate"].apply(lambda v: f"{v * 100:.1f}%")
            disp_prof["avg_predicted_prob"] = disp_prof["avg_predicted_prob"].apply(lambda v: f"{v * 100:.1f}%")
            disp_prof["high_risk_pct"] = disp_prof["high_risk_pct"].apply(lambda v: f"{v * 100:.1f}%")
            disp_prof.columns = [
                "Persona Name",
                "Accounts",
                "Avg Tenure",
                "Avg Bill ($)",
                "Empirical Churn",
                "Model Pred Prob",
                "High Risk %",
            ]
            st.dataframe(disp_prof, use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-top: 14px;">
                <div style="font-weight: 700; color: #0F172A; font-size: 0.85rem; margin-bottom: 4px;">Strategic Summary</div>
                <div style="font-size: 0.8rem; color: #475569; line-height: 1.4;">
                    The <b>Tenured Premium Accounts</b> represent nearly 60% of the customer base and generate the vast majority of MRR,
                    yet exhibit a 32.7% attrition rate. Retention interventions focused on this cohort yield the highest net financial return.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
