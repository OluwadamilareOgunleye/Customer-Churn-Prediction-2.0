"""
Locust load testing scenario for Telco Churn Prediction API.

Usage:
  locust -f tests/performance/locustfile.py --headless -u 20 -r 5 --run-time 30s --host http://localhost:8000
"""

import random
from locust import HttpUser, task, between


SAMPLE_CUSTOMER = {
    "CustomerID": "LOCUST-001",
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
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
    "Monthly Charges": 85.50,
    "Total Charges": 1026.00
}


class ChurnUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def test_health_check(self):
        self.client.get("/health")

    @task(5)
    def test_single_prediction(self):
        payload = SAMPLE_CUSTOMER.copy()
        payload["Tenure Months"] = random.randint(1, 72)
        payload["Monthly Charges"] = round(random.uniform(20.0, 120.0), 2)
        payload["Total Charges"] = round(payload["Tenure Months"] * payload["Monthly Charges"], 2)
        self.client.post("/predict", json=payload)

    @task(1)
    def test_batch_prediction(self):
        batch = []
        for i in range(10):
            item = SAMPLE_CUSTOMER.copy()
            item["CustomerID"] = f"BATCH-{i}"
            batch.append(item)
        self.client.post("/predict/batch", json={"records": batch})
