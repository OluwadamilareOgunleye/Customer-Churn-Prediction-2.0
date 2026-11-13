"""Pydantic schemas for request validation and response serialization."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CustomerRecordInput(BaseModel):
    """Schema representing a single telecommunications customer record."""

    CustomerID: Optional[str] = Field(None, description="Optional customer account identifier")
    Gender: str = Field(..., description="Customer gender")
    Senior_Citizen: str = Field(..., alias="Senior Citizen", description="Senior citizen indicator")
    Partner: str = Field(..., description="Partner indicator")
    Dependents: str = Field(..., description="Dependents indicator")
    Tenure_Months: float = Field(..., alias="Tenure Months", ge=0, description="Customer relationship length in months")
    Phone_Service: str = Field(..., alias="Phone Service", description="Phone service subscription")
    Multiple_Lines: str = Field(..., alias="Multiple Lines", description="Multiple lines status")
    Internet_Service: str = Field(..., alias="Internet Service", description="Internet service type")
    Online_Security: str = Field(..., alias="Online Security", description="Online security add-on")
    Online_Backup: str = Field(..., alias="Online Backup", description="Online backup add-on")
    Device_Protection: str = Field(..., alias="Device Protection", description="Device protection plan")
    Tech_Support: str = Field(..., alias="Tech Support", description="Technical support subscription")
    Streaming_TV: str = Field(..., alias="Streaming TV", description="Streaming TV subscription")
    Streaming_Movies: str = Field(..., alias="Streaming Movies", description="Streaming movies subscription")
    Contract: str = Field(..., description="Contract duration")
    Paperless_Billing: str = Field(..., alias="Paperless Billing", description="Paperless billing preference")
    Payment_Method: str = Field(..., alias="Payment Method", description="Billing payment method")
    Monthly_Charges: float = Field(..., alias="Monthly Charges", ge=0, description="Monthly charges billed in USD")
    Total_Charges: float = Field(..., alias="Total Charges", ge=0, description="Total charges billed to date in USD")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "CustomerID": "CUST-7043",
                "Gender": "Female",
                "Senior Citizen": "No",
                "Partner": "No",
                "Dependents": "No",
                "Tenure Months": 3,
                "Phone Service": "Yes",
                "Multiple Lines": "No",
                "Internet Service": "Fiber optic",
                "Online Security": "No",
                "Online Backup": "No",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "Yes",
                "Streaming Movies": "Yes",
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Electronic check",
                "Monthly Charges": 89.85,
                "Total Charges": 269.55,
            }
        },
    }

    def to_dict_features(self) -> Dict[str, Any]:
        """Convert Pydantic model to dictionary with canonical column names."""
        return {
            "CustomerID": self.CustomerID,
            "Gender": self.Gender,
            "Senior Citizen": self.Senior_Citizen,
            "Partner": self.Partner,
            "Dependents": self.Dependents,
            "Tenure Months": self.Tenure_Months,
            "Phone Service": self.Phone_Service,
            "Multiple Lines": self.Multiple_Lines,
            "Internet Service": self.Internet_Service,
            "Online Security": self.Online_Security,
            "Online Backup": self.Online_Backup,
            "Device Protection": self.Device_Protection,
            "Tech Support": self.Tech_Support,
            "Streaming TV": self.Streaming_TV,
            "Streaming Movies": self.Streaming_Movies,
            "Contract": self.Contract,
            "Paperless Billing": self.Paperless_Billing,
            "Payment Method": self.Payment_Method,
            "Monthly Charges": self.Monthly_Charges,
            "Total Charges": self.Total_Charges,
        }


class PredictionResponse(BaseModel):
    """Prediction output schema for an individual customer."""

    customer_id: Optional[str] = None
    churn_probability: float = Field(..., description="Estimated churn probability in [0, 1]")
    churn_prediction: int = Field(..., description="Binary churn prediction (0: Retained, 1: Churned)")
    churn_label: str = Field(..., description="Categorical label ('No' or 'Yes')")
    risk_category: str = Field(..., description="Assigned risk tier ('Low Risk', 'Medium Risk', 'High Risk')")
    decision_threshold: float = Field(..., description="Classification threshold applied")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal validation or imputation warnings")
    calibration_note: str = Field(..., description="Contextual explanation of probability calibration")


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""

    records: List[CustomerRecordInput]


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""

    total_records: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Service liveness probe response."""

    status: str
    model_loaded: bool
    version: str
    app_name: str


class ReadinessResponse(BaseModel):
    """Service readiness probe response."""

    ready: bool
    status: str
    model_loaded: bool
    decision_threshold: Optional[float] = None
    available_memory_mb: float
    issues: List[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    """Non-sensitive model metadata response."""

    model_name: str
    model_type: str
    version: str
    decision_threshold: float
    feature_count: int
    features: List[str]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
