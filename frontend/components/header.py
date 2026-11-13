"""
Top Navigation & Status Header Component.
"""

from datetime import datetime, timezone
import streamlit as st


def render_header(total_records: int, last_updated: str = None):
    """Render SaaS-style top header with dataset health and refresh trigger."""
    if last_updated is None:
        last_updated = datetime.now(timezone.utc).strftime("%b %d, %Y • %H:%M UTC")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            f"""
            <div style="margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.75rem;">📊</span>
                    <h1 style="font-size: 1.65rem; font-weight: 800; color: #0F172A; margin: 0; letter-spacing: -0.02em;">
                        Customer Churn Intelligence
                    </h1>
                </div>
                <p style="color: #64748B; font-size: 0.9rem; margin: 4px 0 0 42px;">
                    Enterprise subscriber retention, behavioral segmentation & revenue analytics
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div style="text-align: right; margin-top: 4px;">
                <span style="background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">
                    🟢 {total_records:,} Accounts Active
                </span>
                <p style="color: #94A3B8; font-size: 0.72rem; margin: 6px 0 0 0;">
                    Data Sync: {last_updated}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
