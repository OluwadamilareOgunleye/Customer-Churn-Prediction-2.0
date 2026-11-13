"""
Interactive Plotly Visualizations for Customer Churn Intelligence Dashboard.
Adheres to SaaS analytics design principles: high contrast, readable legends, formatted tooltips.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List

# Plotly Color Tokens
PALETTE_PRIMARY = ["#2563EB", "#EF4444", "#10B981", "#F59E0B", "#6366F1", "#8B5CF6"]
COLOR_RETAINED = "#10B981"  # Emerald
COLOR_CHURNED = "#EF4444"   # Red / Coral
COLOR_BAR = "#3B82F6"       # Blue
COLOR_BG = "rgba(0,0,0,0)"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLOR_BG,
    plot_bgcolor=COLOR_BG,
    font=dict(family="Inter, sans-serif", size=12, color="#334155"),
    margin=dict(l=20, r=20, t=40, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hoverlabel=dict(bgcolor="#0F172A", font_size=12, font_family="Inter, sans-serif", font_color="#FFFFFF")
)


def chart_churn_by_tenure_group(df: pd.DataFrame) -> go.Figure:
    """Chart 1: Customer Churn Rates across Tenure Groups."""
    summary = df.groupby("Tenure Group", observed=False)["Churn Value"].agg(
        Total="count",
        Churned=lambda x: (x == 1).sum()
    ).reset_index()
    summary["Churn Rate (%)"] = (summary["Churned"] / summary["Total"]) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["Tenure Group"],
        y=summary["Churn Rate (%)"],
        text=summary["Churn Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        marker_color=["#EF4444" if v >= 35 else "#F59E0B" if v >= 20 else "#10B981" for v in summary["Churn Rate (%)"]],
        hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<br>Total Customers: %{customdata[0]:,}<extra></extra>",
        customdata=summary[["Total"]].values
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Customer Churn Rate by Tenure Group</b>",
        yaxis=dict(title="Churn Rate (%)", range=[0, max(summary["Churn Rate (%)"].max() * 1.25, 50)], gridcolor="#F1F5F9"),
        xaxis=dict(title=None, showgrid=False)
    )
    return fig


def chart_churn_by_living_situation(df: pd.DataFrame) -> go.Figure:
    """Chart 2: Churn by Living Situation (Household Composition)."""
    summary = df.groupby("Living Situation", observed=False)["Churn Value"].agg(
        Total="count",
        Churned=lambda x: (x == 1).sum()
    ).reset_index()
    summary["Churn Rate (%)"] = (summary["Churned"] / summary["Total"]) * 100.0
    summary = summary.sort_values(by="Churn Rate (%)", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=summary["Living Situation"],
        x=summary["Churn Rate (%)"],
        orientation="h",
        text=summary["Churn Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        marker_color="#2563EB",
        hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.1f}%<br>Cohort Size: %{customdata[0]:,}<extra></extra>",
        customdata=summary[["Total"]].values
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Churn Rate by Household Composition</b>",
        xaxis=dict(title="Churn Rate (%)", range=[0, max(summary["Churn Rate (%)"].max() * 1.25, 45)], gridcolor="#F1F5F9"),
        yaxis=dict(title=None, showgrid=False)
    )
    return fig


def chart_revenue_by_contract(df: pd.DataFrame) -> go.Figure:
    """Chart 3: Revenue Contribution and Churn by Contract Type."""
    summary = df.groupby("Contract", observed=False).agg(
        Total_Revenue=("Monthly Charges", "sum"),
        Churn_Rate=("Churn Value", lambda x: (x == 1).mean() * 100.0),
        Total_Count=("Churn Value", "count")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["Contract"],
        y=summary["Total_Revenue"],
        name="Monthly Revenue ($)",
        marker_color="#3B82F6",
        text=summary["Total_Revenue"].apply(lambda v: f"${v:,.0f}"),
        textposition="inside",
        hovertemplate="<b>%{x}</b><br>Total Monthly Revenue: $%{y:,.2f}<br>Subscribers: %{customdata[0]:,}<extra></extra>",
        customdata=summary[["Total_Count"]].values
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Monthly Revenue by Contract Type</b>",
        yaxis=dict(title="Monthly Revenue (USD)", gridcolor="#F1F5F9"),
        xaxis=dict(title=None, showgrid=False)
    )
    return fig


def chart_churn_by_payment_method(df: pd.DataFrame) -> go.Figure:
    """Chart 4: Churn Rate by Billing / Payment Method."""
    summary = df.groupby("Payment Method", observed=False)["Churn Value"].agg(
        Total="count",
        Churned=lambda x: (x == 1).sum()
    ).reset_index()
    summary["Churn Rate (%)"] = (summary["Churned"] / summary["Total"]) * 100.0
    summary = summary.sort_values(by="Churn Rate (%)", ascending=False)

    colors = ["#EF4444" if "Electronic check" in m else "#3B82F6" for m in summary["Payment Method"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["Payment Method"],
        y=summary["Churn Rate (%)"],
        text=summary["Churn Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<br>Accounts: %{customdata[0]:,}<extra></extra>",
        customdata=summary[["Total"]].values
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Churn Rate by Payment Method</b>",
        yaxis=dict(title="Churn Rate (%)", range=[0, 55], gridcolor="#F1F5F9"),
        xaxis=dict(title=None, showgrid=False, tickangle=-15)
    )
    return fig


def chart_churn_by_services_count(df: pd.DataFrame) -> go.Figure:
    """Chart 5: Relationship between Number of Add-on Services and Churn."""
    summary = df.groupby("Service Count", observed=False)["Churn Value"].agg(
        Total="count",
        Churned=lambda x: (x == 1).sum()
    ).reset_index()
    summary["Churn Rate (%)"] = (summary["Churned"] / summary["Total"]) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary["Service Count"],
        y=summary["Churn Rate (%)"],
        mode="lines+markers+text",
        text=summary["Churn Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="top center",
        line=dict(color="#2563EB", width=3),
        marker=dict(size=9, color="#1D4ED8"),
        hovertemplate="<b>%{x} Add-on Services</b><br>Churn Rate: %{y:.1f}%<br>Customers: %{customdata[0]:,}<extra></extra>",
        customdata=summary[["Total"]].values
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Churn Rate vs Count of Add-on Services</b>",
        xaxis=dict(title="Number of Optional Add-on Services (0–6)", dtick=1, showgrid=False),
        yaxis=dict(title="Churn Rate (%)", range=[0, 55], gridcolor="#F1F5F9")
    )
    return fig


def chart_charges_by_churn_status(df: pd.DataFrame) -> go.Figure:
    """Chart 6: Monthly Charges Comparison between Churned vs Retained."""
    fig = px.box(
        df,
        x="Churn Label",
        y="Monthly Charges",
        color="Churn Label",
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED},
        points=False
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Monthly Bill Distribution by Churn Status</b>",
        yaxis=dict(title="Monthly Charges ($ USD)", gridcolor="#F1F5F9"),
        xaxis=dict(title=None, showgrid=False),
        showlegend=False
    )
    return fig


def chart_churn_by_internet_service(df: pd.DataFrame) -> go.Figure:
    """Chart 7: Churn Rates across Internet Service Types."""
    summary = df.groupby("Internet Service", observed=False)["Churn Value"].agg(
        Total="count",
        Churned=lambda x: (x == 1).sum()
    ).reset_index()
    summary["Churn Rate (%)"] = (summary["Churned"] / summary["Total"]) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["Internet Service"],
        y=summary["Churn Rate (%)"],
        text=summary["Churn Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        marker_color=["#3B82F6", "#EF4444", "#10B981"],
        hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<br>Subscribers: %{customdata[0]:,}<extra></extra>",
        customdata=summary[["Total"]].values
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Churn Rate by Internet Service Type</b>",
        yaxis=dict(title="Churn Rate (%)", range=[0, 50], gridcolor="#F1F5F9"),
        xaxis=dict(title=None, showgrid=False)
    )
    return fig


def chart_contract_retention_stacked(df: pd.DataFrame) -> go.Figure:
    """Chart 8: Contract Duration and Customer Retention Split."""
    summary = df.groupby(["Contract", "Churn Label"], observed=False).size().unstack(fill_value=0).reset_index()
    if "Retained" not in summary: summary["Retained"] = 0
    if "Churned" not in summary: summary["Churned"] = 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Retained",
        x=summary["Contract"],
        y=summary["Retained"],
        marker_color=COLOR_RETAINED,
        hovertemplate="<b>%{x}</b><br>Retained: %{y:,}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        name="Churned",
        x=summary["Contract"],
        y=summary["Churned"],
        marker_color=COLOR_CHURNED,
        hovertemplate="<b>%{x}</b><br>Churned: %{y:,}<extra></extra>"
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        barmode="stack",
        title="<b>Customer Retention & Churn by Contract</b>",
        yaxis=dict(title="Number of Customers", gridcolor="#F1F5F9"),
        xaxis=dict(title=None, showgrid=False)
    )
    return fig


def chart_tenure_distribution(df: pd.DataFrame) -> go.Figure:
    """Chart 9: Overall Customer Base Tenure Distribution."""
    fig = px.histogram(
        df,
        x="Tenure Months",
        nbins=24,
        color_discrete_sequence=["#3B82F6"]
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Overall Customer Tenure Distribution</b>",
        xaxis=dict(title="Tenure (Months)", dtick=12, gridcolor="#F1F5F9"),
        yaxis=dict(title="Customer Count", gridcolor="#F1F5F9"),
        showlegend=False
    )
    return fig


def chart_risk_distribution(risk_counts: Dict[str, int]) -> go.Figure:
    """Chart 10: Model Calibrated Customer Risk Distribution Tiers."""
    tier_order = ["Low Risk", "Moderate Risk", "Elevated Risk", "High Risk"]
    counts = [risk_counts.get(tier, 0) for tier in tier_order]
    colors = ["#10B981", "#3B82F6", "#F59E0B", "#EF4444"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tier_order,
        y=counts,
        text=[f"{c:,}" for c in counts],
        textposition="outside",
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Accounts: %{y:,}<extra></extra>"
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Customer Risk Distribution (Calibrated Model Tiers)</b>",
        yaxis=dict(title="Accounts in Tier", gridcolor="#F1F5F9"),
        xaxis=dict(title=None, showgrid=False)
    )
    return fig
