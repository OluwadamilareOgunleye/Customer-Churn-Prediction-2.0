"""
Global Sidebar Filters Component.
Manages interactive slice selection across all analytics pages.
"""

from typing import Dict, Any
import streamlit as st


def render_sidebar_filters(df) -> Dict[str, Any]:
    """Render structured sidebar filters and return user choices."""
    st.sidebar.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">
                Global Cohort Filters
            </h3>
            <p style="font-size: 0.78rem; color: #64748B; margin: 2px 0 0 0;">
                Slice dashboard metrics dynamically
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Reset button
    if st.sidebar.button("🔄 Reset Filters", use_container_width=True):
        st.session_state["filter_gender"] = "All"
        st.session_state["filter_senior"] = "All"
        st.session_state["filter_contract"] = "All"
        st.session_state["filter_internet"] = "All"
        st.session_state["filter_payment"] = "All"
        st.session_state["filter_tenure"] = "All"
        st.session_state["filter_churn"] = "All"
        st.rerun()

    # Initial session states
    defaults = {
        "filter_gender": "All",
        "filter_senior": "All",
        "filter_contract": "All",
        "filter_internet": "All",
        "filter_payment": "All",
        "filter_tenure": "All",
        "filter_churn": "All",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    contract_opts = ["All"] + sorted([str(x) for x in df["Contract"].dropna().unique()])
    internet_opts = ["All"] + sorted([str(x) for x in df["Internet Service"].dropna().unique()])
    payment_opts = ["All"] + sorted([str(x) for x in df["Payment Method"].dropna().unique()])
    tenure_opts = ["All", "0–12 Mos", "13–24 Mos", "25–48 Mos", "49–72 Mos"]
    gender_opts = ["All", "Male", "Female"]
    senior_opts = ["All", "Senior", "Non-Senior"]
    churn_opts = ["All", "Retained", "Churned"]

    selected_contract = st.sidebar.selectbox("Contract Type", contract_opts, key="filter_contract")
    selected_internet = st.sidebar.selectbox("Internet Service", internet_opts, key="filter_internet")
    selected_tenure = st.sidebar.selectbox("Tenure Group", tenure_opts, key="filter_tenure")
    selected_payment = st.sidebar.selectbox("Payment Method", payment_opts, key="filter_payment")

    with st.sidebar.expander("More Demographics & Status", expanded=False):
        selected_gender = st.selectbox("Gender", gender_opts, key="filter_gender")
        selected_senior = st.selectbox("Senior Status", senior_opts, key="filter_senior")
        selected_churn = st.selectbox("Churn Status", churn_opts, key="filter_churn")

    active_count = sum([
        selected_contract != "All",
        selected_internet != "All",
        selected_tenure != "All",
        selected_payment != "All",
        selected_gender != "All",
        selected_senior != "All",
        selected_churn != "All",
    ])

    if active_count > 0:
        st.sidebar.caption(f"⚡ **{active_count} active filter(s) applied**")

    st.sidebar.divider()

    return {
        "gender": selected_gender,
        "senior": selected_senior,
        "contract": selected_contract,
        "internet": selected_internet,
        "payment": selected_payment,
        "tenure_group": selected_tenure,
        "churn_status": selected_churn,
        "active_count": active_count,
    }
