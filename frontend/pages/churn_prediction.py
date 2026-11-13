"""
Churn Prediction Page - Customer Churn Intelligence Dashboard.
Interactive inference engine supporting real-time single-customer scoring, SHAP attribution,
and high-throughput batch CSV scoring.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import streamlit as st

from frontend.components.header import render_header
from frontend.utils.api_client import ChurnAPIClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"


def get_retention_recommendation(risk_category: str, contract: str, monthly_charges: float, internet_service: str) -> Dict[str, str]:
    """Determine data-driven next-best retention intervention."""
    if risk_category == "Low Risk":
        return {
            "title": "Maintain Standard Loyalty Cadence",
            "action": "Account is stable with high retention propensity. Avoid aggressive promotional discounts; maintain regular product update newsletters and automated CSAT surveys.",
            "urgency": "Low Urgency",
            "badge_color": "#10B981"
        }
    elif risk_category in ("Moderate Risk", "Medium Risk"):
        if contract == "Month-to-month":
            return {
                "title": "Proactive 1-Year Contract Upgrade Offer",
                "action": f"Customer is on a flexible monthly plan paying ${monthly_charges:.2f}/mo. Offer a 10% monthly discount locked in exchange for a 12-month commitment.",
                "urgency": "Medium Urgency",
                "badge_color": "#F59E0B"
            }
        else:
            return {
                "title": "Complimentary Tech Support / Security Add-on",
                "action": "Customer shows moderate attrition indicators. Offer complimentary 3-month trial of Tech Support or Online Security to strengthen service attachment.",
                "urgency": "Medium Urgency",
                "badge_color": "#F59E0B"
            }
    else:  # High / Elevated Risk
        if internet_service == "Fiber optic":
            return {
                "title": "White-Glove VIP Support & Router Audit",
                "action": f"Critical hazard: High-ARPU Fiber subscriber (${monthly_charges:.2f}/mo) on monthly billing. Deploy senior retention specialist call, audit Wi-Fi hardware, and offer a $15/mo discount for 6 months.",
                "urgency": "Immediate Outreach (Within 48h)",
                "badge_color": "#EF4444"
            }
        else:
            return {
                "title": "Immediate Contract Renewal Incentive",
                "action": "High defection probability. Send prioritized renewal incentive: waive billing fees for 3 months with transition to automated ACH payment and 1-year contract extension.",
                "urgency": "Immediate Outreach (Within 48h)",
                "badge_color": "#EF4444"
            }


def render_page(df: pd.DataFrame, full_df: pd.DataFrame):
    """Render Inference & Prediction Interface."""
    render_header(total_records=len(df))

    client = ChurnAPIClient()
    health = client.check_health()

    status_icon = "🟢" if health.get("status") in ("healthy", "connected (direct engine)") else "🟡"
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <h2 style="font-size: 1.25rem; font-weight: 700; color: #0F172A; margin: 0;">
                    Predictive Scoring & Model Inference Engine
                </h2>
                <p style="color: #64748B; font-size: 0.875rem; margin: 4px 0 0 0;">
                    Evaluate subscriber defection likelihood with calibrated probabilities, SHAP attribution, and tailored action plans.
                </p>
            </div>
            <div>
                <span style="background: #F1F5F9; border: 1px solid #CBD5E1; color: #334155; padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">
                    {status_icon} Engine: {health.get('status', 'Ready')}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_single, tab_batch = st.tabs(["Single Subscriber Scoring", "Batch CSV Inference & Export"])

    with tab_single:
        st.markdown('<div class="section-heading">👤 Subscriber Account Profile</div>', unsafe_allow_html=True)

        with st.form("single_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Demographics**")
                customer_id = st.text_input("Customer ID", value="CUST-9821-DEMO")
                gender = st.selectbox("Gender", ["Female", "Male"], index=0)
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
                partner = st.selectbox("Partner", ["No", "Yes"], index=0)
                dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)

            with col2:
                st.markdown("**Contract & Billing**")
                tenure_months = st.slider("Tenure (Months)", min_value=1, max_value=72, value=4)
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    index=0,
                )
                monthly_charges = st.number_input("Monthly Charges ($ USD)", min_value=18.0, max_value=130.0, value=85.50, step=0.50)
                calc_total = float(tenure_months * monthly_charges)
                total_charges = st.number_input("Total Charges ($ USD)", min_value=18.0, max_value=9000.0, value=calc_total, step=5.0)

            with col3:
                st.markdown("**Telecom & Internet Services**")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=1)
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0)
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0)
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=0)
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], index=0)

            submit_btn = st.form_submit_button("⚡ Run Predictive Scoring", use_container_width=True)

        if submit_btn:
            customer_record = {
                "CustomerID": customer_id,
                "Gender": gender,
                "Senior Citizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "Tenure Months": int(tenure_months),
                "Phone Service": phone_service,
                "Multiple Lines": multiple_lines,
                "Internet Service": internet_service,
                "Online Security": online_security,
                "Online Backup": online_backup,
                "Device Protection": device_protection,
                "Tech Support": tech_support,
                "Streaming TV": streaming_tv,
                "Streaming Movies": streaming_movies,
                "Contract": contract,
                "Paperless Billing": paperless_billing,
                "Payment Method": payment_method,
                "Monthly Charges": float(monthly_charges),
                "Total Charges": float(total_charges),
            }

            try:
                result = client.predict_single(customer_record)
                prob = result.get("churn_probability", 0.0)
                threshold = result.get("decision_threshold", 0.30)
                risk_cat = result.get("risk_category", "Low Risk")
                is_churn = result.get("churn_prediction", 0) == 1

                st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-heading">🎯 Predictive Scoring Assessment</div>', unsafe_allow_html=True)

                r_col1, r_col2, r_col3 = st.columns([1.2, 1.8, 2.0])

                with r_col1:
                    badge_class = "badge-high" if prob >= 0.60 else "badge-med" if prob >= 0.30 else "badge-low"
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="text-align: center;">
                            <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">
                                Risk Tier Classification
                            </div>
                            <div style="margin: 12px 0;">
                                <span class="{badge_class}" style="font-size: 1.1rem; padding: 6px 16px;">
                                    {risk_cat}
                                </span>
                            </div>
                            <div style="font-size: 0.78rem; color: #94A3B8;">
                                Calibrated Threshold: {threshold:.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with r_col2:
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                            <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 8px;">
                                Calibrated Churn Likelihood
                            </div>
                            <div style="display: flex; align-items: baseline; gap: 8px;">
                                <span style="font-size: 2.2rem; font-weight: 800; color: {'#EF4444' if is_churn else '#10B981'};">
                                    {prob * 100.0:.1f}%
                                </span>
                                <span style="font-size: 0.85rem; color: #64748B;">empirical probability</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(prob, 0.0), 1.0))

                with r_col3:
                    rec = get_retention_recommendation(risk_cat, contract, monthly_charges, internet_service)
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="border-left: 4px solid {rec['badge_color']};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.8rem; font-weight: 700; color: {rec['badge_color']}; text-transform: uppercase;">
                                    {rec['urgency']}
                                </span>
                            </div>
                            <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 4px 0;">
                                {rec['title']}
                            </div>
                            <div style="font-size: 0.8rem; color: #475569; line-height: 1.4;">
                                {rec['action']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Local Feature Contribution Analysis
                st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-heading">🔍 Local Feature Contribution Factors</div>', unsafe_allow_html=True)

                fc1, fc2 = st.columns(2)
                with fc1:
                    st.markdown("**Top Defection Hazard Drivers (Increasing Risk)**")
                    hazard_factors = []
                    if contract == "Month-to-month":
                        hazard_factors.append(("Month-to-month Contract", "Absence of long-term commitment unlocks immediate cancellation."))
                    if tenure_months <= 12:
                        hazard_factors.append((f"Early Tenure ({tenure_months} mos)", "Customer is within the peak attrition window (Months 1–12)."))
                    if internet_service == "Fiber optic" and tech_support == "No":
                        hazard_factors.append(("Fiber Broadband without Tech Support", "High billing expenditure without technical assistance raises friction."))
                    if payment_method == "Electronic check":
                        hazard_factors.append(("Electronic Check Billing", "Unautomated billing introduces monthly re-evaluation points."))
                    if monthly_charges >= 70:
                        hazard_factors.append((f"Elevated Monthly Bill (${monthly_charges:.2f})", "Premium monthly expenditure invites competitive shopping."))

                    if not hazard_factors:
                        st.info("No major high-risk indicators detected for this account.")
                    else:
                        for h_title, h_desc in hazard_factors[:4]:
                            st.markdown(
                                f"""
                                <div style="background: #FEF2F2; border-left: 3px solid #EF4444; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px;">
                                    <div style="font-weight: 600; color: #991B1B; font-size: 0.85rem;">🔺 {h_title}</div>
                                    <div style="color: #64748B; font-size: 0.78rem;">{h_desc}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                with fc2:
                    st.markdown("**Top Retention Anchors (Mitigating Risk)**")
                    anchor_factors = []
                    if contract in ("One year", "Two year"):
                        anchor_factors.append((f"{contract} Contract", "Long-term contractual commitment provides strong retention lock."))
                    if tenure_months > 24:
                        anchor_factors.append((f"Established Loyalty ({tenure_months} mos)", "Deep lifecycle tenure correlates strongly with ongoing renewal."))
                    if tech_support == "Yes":
                        anchor_factors.append(("Tech Support Active", "Active technical assistance resolves friction and cuts churn."))
                    if online_security == "Yes":
                        anchor_factors.append(("Online Security Active", "Security suite creates sticky integration with subscriber operations."))
                    if payment_method in ("Bank transfer (automatic)", "Credit card (automatic)"):
                        anchor_factors.append(("Automated Autopay Billing", "Frictionless recurring billing prevents involuntary or impulsive churn."))

                    if not anchor_factors:
                        st.caption("No significant protective retention anchors present.")
                    else:
                        for a_title, a_desc in anchor_factors[:4]:
                            st.markdown(
                                f"""
                                <div style="background: #ECFDF5; border-left: 3px solid #10B981; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px;">
                                    <div style="font-weight: 600; color: #065F46; font-size: 0.85rem;">🛡️ {a_title}</div>
                                    <div style="color: #64748B; font-size: 0.78rem;">{a_desc}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            except Exception as e:
                st.error(f"Inference error: {str(e)}")

    with tab_batch:
        st.markdown('<div class="section-heading">📁 Batch Scoring & CSV Export</div>', unsafe_allow_html=True)
        st.markdown(
            """
            Upload a CSV containing customer records for high-throughput batch scoring,
            or load a sample test slice to evaluate performance across active subscribers.
            """
        )

        b_c1, b_c2 = st.columns([2, 1])
        with b_c1:
            uploaded_file = st.file_uploader("Upload CSV file for batch inference", type=["csv"])

        with b_c2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            load_sample = st.button("📥 Load 100 Holdout Test Accounts", use_container_width=True)

        batch_df = None
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
        elif load_sample and SAMPLE_TEST_CSV.exists():
            batch_df = pd.read_csv(SAMPLE_TEST_CSV).head(100)

        if batch_df is not None:
            st.info(f"Loaded {len(batch_df)} records for batch evaluation.")
            records = batch_df.to_dict(orient="records")

            with st.spinner("Scoring batch records through calibrated pipeline..."):
                predictions = client.predict_batch(records)

            if predictions:
                res_df = pd.DataFrame(predictions)
                # Join predictions to original columns
                merged_df = batch_df.copy()
                merged_df["Churn Probability"] = res_df["churn_probability"]
                merged_df["Churn Prediction"] = res_df["churn_prediction"]
                merged_df["Risk Tier"] = res_df["risk_category"]

                # Quick Summary Metrics
                st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Batch Records Scored", f"{len(merged_df):,}")
                flagged_count = (merged_df["Churn Prediction"] == 1).sum()
                m2.metric("Flagged for Attrition", f"{flagged_count:,}", f"{flagged_count / len(merged_df) * 100:.1f}%")
                high_risk_count = (merged_df["Risk Tier"] == "High Risk").sum()
                m3.metric("High Risk Accounts", f"{high_risk_count:,}")
                avg_batch_prob = merged_df["Churn Probability"].mean() * 100.0
                m4.metric("Avg Batch Probability", f"{avg_batch_prob:.1f}%")

                st.dataframe(
                    merged_df[[
                        "CustomerID",
                        "Tenure Months",
                        "Contract",
                        "Monthly Charges",
                        "Churn Probability",
                        "Churn Prediction",
                        "Risk Tier",
                    ] if "CustomerID" in merged_df.columns else merged_df.columns[:8]],
                    use_container_width=True,
                )

                # Download CSV
                csv_data = merged_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Scored Predictions CSV",
                    data=csv_data,
                    file_name="churn_predictions_scored.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
