"""
Model Performance Page - Customer Churn Intelligence Dashboard.
Displays validated holdout test-set metrics, calibration curves, confusion matrices, and threshold trade-offs.
"""

from pathlib import Path
import streamlit as st

from frontend.components.header import render_header

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def render_page(df=None, full_df=None):
    """Render Production Model Evaluation & Diagnostics."""
    render_header(total_records=1409)

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="font-size: 1.25rem; font-weight: 700; color: #0F172A; margin: 0;">
                Production Model Evaluation & Generalization Diagnostics
            </h2>
            <p style="color: #64748B; font-size: 0.875rem; margin: 4px 0 0 0;">
                Tuned <b>HistGradientBoostingClassifier</b> with Isotonic Probability Calibration evaluated on 1,409 untouched holdout test records.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5 KPI Metric Cards for Model Performance
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">ROC-AUC</div>
                <div class="kpi-value" style="color: #2563EB;">0.8570</div>
                <div class="kpi-subtext">Overall ranking power</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">PR-AUC</div>
                <div class="kpi-value" style="color: #0D9488;">0.6740</div>
                <div class="kpi-subtext">Precision-recall tradeoff</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Recall (Sensitivity)</div>
                <div class="kpi-value" style="color: #10B981;">77.01%</div>
                <div class="kpi-subtext">288 / 374 churners caught</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Precision</div>
                <div class="kpi-value" style="color: #F59E0B;">54.24%</div>
                <div class="kpi-subtext">54% flagged are churners</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m5:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Brier Score</div>
                <div class="kpi-value" style="color: #6366F1;">0.1322</div>
                <div class="kpi-subtext">Calibrated probability error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)

    # Visual Diagnostics Grid - Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-heading">📈 ROC & Precision-Recall Curves</div>', unsafe_allow_html=True)
        img_roc = FIGURES_DIR / "12_test_roc_pr_curves.png"
        if img_roc.exists():
            st.image(str(img_roc), use_container_width=True, caption="Figure 12: Holdout Test ROC and PR Curves")

    with c2:
        st.markdown('<div class="section-heading">🎯 Test Confusion Matrix (Threshold = 0.30)</div>', unsafe_allow_html=True)
        img_cm = FIGURES_DIR / "10_test_confusion_matrix_calibrated.png"
        if img_cm.exists():
            st.image(str(img_cm), use_container_width=True, caption="Figure 10: Confusion Matrix on Test Set at Calibrated Threshold")

    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

    # Visual Diagnostics Grid - Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-heading">⚖️ Probability Calibration Curve (Reliability)</div>', unsafe_allow_html=True)
        img_calib = FIGURES_DIR / "18_calibration_curve_comparison.png"
        if img_calib.exists():
            st.image(str(img_calib), use_container_width=True, caption="Figure 18: Reliability Curve — Uncalibrated vs Isotonic Calibrated")

    with c4:
        st.markdown('<div class="section-heading">🎚️ Threshold vs Precision / Recall Trade-off</div>', unsafe_allow_html=True)
        img_thresh = FIGURES_DIR / "19_threshold_tradeoff_curve.png"
        if img_thresh.exists():
            st.image(str(img_thresh), use_container_width=True, caption="Figure 19: Metric Tradeoffs across Decision Thresholds (0.05 to 0.95)")

    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)

    # Technical Rationale & Audit Summary
    st.markdown('<div class="section-heading">🛡️ Engineering Rationale & Calibration Strategy</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 18px;">
            <h4 style="margin: 0 0 8px 0; color: #0F172A; font-size: 1rem;">Why the Calibrated 0.30 Decision Threshold?</h4>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin: 0 0 10px 0;">
                In subscriber retention, false negatives (missing a customer who defects, losing ~$1,200 lifetime revenue)
                are roughly 8x more expensive than false positives (reaching out with a $50 incentive to an account that would have stayed).
                Operating at the uncalibrated 0.50 threshold yields 52.7% recall, abandoning nearly half of churners.
                Calibrating the threshold to <b>0.30</b> lifts sensitivity to <b>77.01%</b>, recovering 288 out of 374 actual test-set churners
                and producing peak net business profitability ($59,850 net profit; 225.4% ROI).
            </p>
            <div style="display: flex; gap: 20px; font-size: 0.8rem; color: #64748B;">
                <span>• <b>Framework:</b> Scikit-Learn 1.6+ Pipeline</span>
                <span>• <b>Cross-Validation:</b> 5-Fold Stratified CV</span>
                <span>• <b>Calibration Method:</b> Isotonic Regression</span>
                <span>• <b>Test Size:</b> 1,409 Records (20% holdout)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
