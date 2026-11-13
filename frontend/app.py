"""
Customer Churn Intelligence - Enterprise Analytics & Retention Optimization Platform.
Production Streamlit Application implementing modern SaaS UX and data-driven intelligence.
"""

from pathlib import Path
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Enterprise Theme & CSS
from frontend.theme import CUSTOM_CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 3. Load Data & Components
from frontend.utils.data_loader import load_full_dataset, apply_global_filters
from frontend.components.filters import render_sidebar_filters
from frontend.utils.api_client import ChurnAPIClient

# Load Full Dataset (7,043 accounts)
try:
    full_df = load_full_dataset()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# 4. Sidebar Navigation
st.sidebar.markdown(
    """
    <div style="padding: 4px 0 16px 0;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.5rem;">📊</span>
            <span style="font-weight: 800; font-size: 1.15rem; color: #0F172A; letter-spacing: -0.02em;">
                Churn Intelligence
            </span>
        </div>
        <div style="font-size: 0.72rem; color: #64748B; margin-top: 2px;">
            IBM Telco Retention Platform • v1.0
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

NAV_PAGES = [
    "1. Executive Overview",
    "2. Customer Analytics",
    "3. Churn Analysis",
    "4. Churn Prediction",
    "5. Customer Segmentation",
    "6. Model Performance",
    "7. Business Insights & ROI",
]

selected_page = st.sidebar.radio(
    "Navigation View",
    NAV_PAGES,
    index=0,
    label_visibility="collapsed",
)

st.sidebar.divider()

# 5. Global Sidebar Filters
selected_filters = render_sidebar_filters(full_df)
filtered_df = apply_global_filters(full_df, selected_filters)

st.sidebar.divider()

# Sidebar Engine & Health Footer
client = ChurnAPIClient()
health = client.check_health()
status_icon = "🟢" if health.get("status") in ("healthy", "connected (direct engine)") else "🟡"

st.sidebar.markdown(
    f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; margin-top: 10px;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 4px;">
            System Status
        </div>
        <div style="font-size: 0.82rem; color: #0F172A; font-weight: 600;">
            {status_icon} <code>{health.get('status', 'Ready')}</code>
        </div>
        <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 2px;">
            Serving: HistGradientBoosting + Isotonic
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 6. Page Routing
if selected_page == "1. Executive Overview":
    from frontend.pages.executive_overview import render_page
    render_page(filtered_df, full_df)

elif selected_page == "2. Customer Analytics":
    from frontend.pages.customer_analytics import render_page
    render_page(filtered_df, full_df)

elif selected_page == "3. Churn Analysis":
    from frontend.pages.churn_analysis import render_page
    render_page(filtered_df, full_df)

elif selected_page == "4. Churn Prediction":
    from frontend.pages.churn_prediction import render_page
    render_page(filtered_df, full_df)

elif selected_page == "5. Customer Segmentation":
    from frontend.pages.segmentation import render_page
    render_page(filtered_df, full_df)

elif selected_page == "6. Model Performance":
    from frontend.pages.model_performance import render_page
    render_page(filtered_df, full_df)

elif selected_page == "7. Business Insights & ROI":
    from frontend.pages.business_roi import render_page
    render_page(filtered_df, full_df)
