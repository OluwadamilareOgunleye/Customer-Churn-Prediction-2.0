# Phase 10: Testing Strategy and Quality Assurance Architecture

## 1. Quality Assurance Philosophy

This project adopts a **multi-tiered testing pyramid** tailored for production machine learning systems:
1. **Unit Testing:** Fast, isolated tests for deterministic utility functions (data cleaning, type casting, schema boundary validation, risk categorization).
2. **Integration Testing:** Tests validating the contract between components (e.g., Scikit-Learn Pipeline and ChurnPredictor, Pydantic schemas and FastAPI route handlers).
3. **End-to-End (E2E) Testing:** Complete lifecycle simulation mirroring production consumer requests (submitting raw customer payloads through HTTP endpoints to receive calibrated churn probabilities).
4. **Data Integrity & Leakage Testing:** Specialized checks asserting zero index overlap and class balance preservation during train/test splitting.

---

## 2. Test Suite Architecture

```
tests/
├── test_data_validation.py    # Schema integrity, whitespace imputation, split isolation
├── test_model_inference.py     # Artifact loading, probability bounds, risk categorization
├── test_api.py                 # FastAPI endpoints, Pydantic validation, status codes
└── test_integration_e2e.py    # Multi-component E2E workflow & client fallback verification
```

---

## 3. Running the Test Suite

### Full Test Suite Run
```powershell
pytest -v
```

### Run with Coverage Reporting
```powershell
pytest --cov=src --cov=api --cov-report=term-missing
```

### Run Targeted Test File
```powershell
pytest tests/test_api.py -v
```
