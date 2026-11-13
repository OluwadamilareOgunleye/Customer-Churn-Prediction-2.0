"""
UI Theme, Color Palette & CSS Styles for Customer Churn Intelligence Dashboard.
Adheres to enterprise SaaS design language (Inter typography, subtle borders, clean contrast).
"""

PRIMARY_NAVY = "#0F172A"       # Slate 900
SECONDARY_BLUE = "#2563EB"     # Blue 600
ACCENT_TEAL = "#0D9488"        # Teal 600
SUCCESS_GREEN = "#10B981"      # Emerald 500
DANGER_RED = "#EF4444"         # Red 500
WARNING_AMBER = "#F59E0B"      # Amber 500
TEXT_DARK = "#1E293B"          # Slate 800
TEXT_MUTED = "#64748B"         # Slate 500
BG_LIGHT = "#F8FAFC"           # Slate 50
CARD_BG = "#FFFFFF"            # White
BORDER_COLOR = "#E2E8F0"       # Slate 200

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1E293B;
}

/* Main Container Padding */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
}

/* Header Container */
.header-container {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 24px;
    color: #FFFFFF;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-title {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
    color: #FFFFFF;
}

.header-subtitle {
    font-size: 0.875rem;
    color: #94A3B8;
    margin: 4px 0 0 0;
}

.header-status-badge {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34D399;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Metric Cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    height: 100%;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
}

.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.kpi-title {
    font-size: 0.8125rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748B;
    margin: 0;
}

.kpi-icon {
    font-size: 1.1rem;
    padding: 6px;
    border-radius: 8px;
    background: #F1F5F9;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #0F172A;
    margin: 0;
    line-height: 1.2;
}

.kpi-subtext {
    font-size: 0.75rem;
    color: #94A3B8;
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}

.kpi-badge-positive {
    color: #10B981;
    font-weight: 600;
}

.kpi-badge-negative {
    color: #EF4444;
    font-weight: 600;
}

.kpi-badge-neutral {
    color: #2563EB;
    font-weight: 600;
}

/* Insight Cards */
.insight-card {
    background: #FFFFFF;
    border-left: 4px solid #2563EB;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    border-top: 1px solid #F1F5F9;
    border-right: 1px solid #F1F5F9;
    border-bottom: 1px solid #F1F5F9;
}

.insight-card.danger {
    border-left-color: #EF4444;
}

.insight-card.success {
    border-left-color: #10B981;
}

.insight-card.warning {
    border-left-color: #F59E0B;
}

.insight-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: #0F172A;
    margin: 0 0 4px 0;
}

.insight-desc {
    font-size: 0.8125rem;
    color: #475569;
    margin: 0 0 6px 0;
    line-height: 1.4;
}

.insight-action {
    font-size: 0.75rem;
    font-weight: 600;
    color: #2563EB;
    background: #EFF6FF;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-block;
}

/* Custom Tabs & Buttons */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 0.875rem;
    color: #64748B;
}

.stTabs [aria-selected="true"] {
    background-color: #EFF6FF !important;
    color: #2563EB !important;
}

/* Section Dividers */
.section-heading {
    font-size: 1.125rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.01em;
    margin-top: 24px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
"""
