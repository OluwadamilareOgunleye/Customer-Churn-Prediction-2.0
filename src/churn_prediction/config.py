"""Project configuration and global constants for Customer Churn Prediction."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco_customer_churn.xlsx"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PHASE_3_REPORTS_DIR = REPORTS_DIR / "phase_3"
PHASE_4_REPORTS_DIR = REPORTS_DIR / "phase_4"
PHASE_5_REPORTS_DIR = REPORTS_DIR / "phase_5"
PHASE_6_REPORTS_DIR = REPORTS_DIR / "phase_6"
PHASE_10_REPORTS_DIR = REPORTS_DIR / "phase_10"
PHASE_13_REPORTS_DIR = REPORTS_DIR / "phase_13"
PHASE_15_REPORTS_DIR = REPORTS_DIR / "phase_15"
PHASE_16_REPORTS_DIR = REPORTS_DIR / "phase_16"
PHASE_17_REPORTS_DIR = REPORTS_DIR / "phase_17"
PHASE_18_REPORTS_DIR = REPORTS_DIR / "phase_18"
PHASE_19_REPORTS_DIR = REPORTS_DIR / "phase_19"
PHASE_20_REPORTS_DIR = REPORTS_DIR / "phase_20"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure runtime directories exist
for directory in [
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    PHASE_3_REPORTS_DIR,
    PHASE_4_REPORTS_DIR,
    PHASE_5_REPORTS_DIR,
    PHASE_6_REPORTS_DIR,
    PHASE_10_REPORTS_DIR,
    PHASE_13_REPORTS_DIR,
    PHASE_15_REPORTS_DIR,
    PHASE_16_REPORTS_DIR,
    PHASE_17_REPORTS_DIR,
    PHASE_18_REPORTS_DIR,
    PHASE_19_REPORTS_DIR,
    PHASE_20_REPORTS_DIR,
    CONFIGS_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed & Splits
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Target Variables
TARGET_COL = "Churn Value"
TARGET_LABEL_COL = "Churn Label"
TARGET_MAPPING = {"No": 0, "Yes": 1}

# Identifiers & Leakage/Unusable Columns
ID_COL = "CustomerID"

# Columns to exclude from predictive features:
# - CustomerID: High cardinality unique identifier
# - Count, Country, State: Constant zero-variance columns
# - City, Zip Code, Lat Long, Latitude, Longitude: Spatial/geographic coordinates
# - Churn Reason: Post-churn exit interview feedback (Target Leakage)
# - Churn Score: IBM's pre-computed churn model output (Target/Model Leakage)
# - CLTV: Post-hoc metric calculated from churn formula
# - Churn Label: String equivalent of target Churn Value
EXCLUDE_COLS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

# Predictive Numerical Features
NUMERICAL_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

# Predictive Categorical Features
CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

# All expected predictive features
ALL_FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
