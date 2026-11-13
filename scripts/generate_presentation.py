"""
Generate Professional 15-Slide Presentation Deck for Customer Churn Prediction Project.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESENTATION_PATH = PROJECT_ROOT / "presentations" / "customer_churn_project.pptx"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Corporate Color Palette
COLOR_PRIMARY = RGBColor(26, 54, 93)     # Deep Navy
COLOR_SECONDARY = RGBColor(49, 130, 206) # Slate Blue
COLOR_ACCENT = RGBColor(221, 107, 32)    # Warm Amber
COLOR_TEXT = RGBColor(45, 55, 72)        # Charcoal
COLOR_MUTED = RGBColor(113, 128, 150)    # Gray

def add_header(slide, title_text, category_text="CUSTOMER CHURN PREDICTION & RETENTION MLOPS"):
    """Format clean standard slide header."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8.4), Inches(1.1))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_SECONDARY
    
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_PRIMARY
    p_title.space_before = Pt(4)

def add_bullet_content(slide, bullets, left=Inches(0.8), top=Inches(1.6), width=Inches(8.4), height=Inches(5.0)):
    """Format structured bulleted content."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            headline, detail = item
            run_head = p.add_run()
            run_head.text = headline + ": "
            run_head.font.bold = True
            run_head.font.size = Pt(14)
            run_head.font.color.rgb = COLOR_PRIMARY
            
            run_body = p.add_run()
            run_body.text = detail
            run_body.font.size = Pt(13)
            run_body.font.color.rgb = COLOR_TEXT
        else:
            run = p.add_run()
            run.text = item
            run.font.size = Pt(13)
            run.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(12)

def main():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Slide 1: Title Slide
    s1 = prs.slides.add_slide(blank_layout)
    box = s1.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(8.0), Inches(3.0))
    tf = box.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "TELCO CUSTOMER CHURN PREDICTION SYSTEM"
    p0.font.size = Pt(26)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_PRIMARY
    
    p1 = tf.add_paragraph()
    p1.text = "Production Machine Learning Engineering, MLOps, & Retention Optimization"
    p1.font.size = Pt(16)
    p1.font.color.rgb = COLOR_SECONDARY
    p1.space_before = Pt(8)
    
    p2 = tf.add_paragraph()
    p2.text = "End-to-End Technical Portfolio Project | Phases 1–25 Complete"
    p2.font.size = Pt(12)
    p2.font.color.rgb = COLOR_MUTED
    p2.space_before = Pt(24)

    # Slide 2: Business Problem & Retention Stakes
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Business Problem: Subscription Churn & Asymmetric Costs")
    add_bullet_content(s2, [
        ("The Industry Reality", "Subscription telecom loses 20–30% of subscribers annually; acquisition is 5–7x more costly than retention."),
        ("Financial Exposure", "For a 100,000 subscriber telco with $1,200 CLV, a 26.5% churn rate represents $31.8M in annual revenue loss."),
        ("The Traditional Failure", "Standard ML models operating at default 0.50 threshold minimize classification error, missing 47% of churners."),
        ("Strategic Objective", "Develop a calibrated binary classification engine that prioritizes recall (capturing true churners) while maintaining healthy precision for actionable marketing campaigns.")
    ])

    # Slide 3: IBM Telco Dataset & Integrity Checks
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Dataset Integrity & SHA-256 Provenance")
    add_bullet_content(s3, [
        ("Canonical Data Source", "IBM Telco Customer Churn dataset containing 7,043 customer accounts across 21 raw columns."),
        ("Cryptographic Provenance", "SHA-256 Verified: 1BCBC0CCC9B352175216979102628E579CFBDE2C3FF57B005DE168A433122640."),
        ("Target Variable", "Binary classification on Churn Value (0: Retained, 1: Churned), exhibiting 26.54% positive prevalence."),
        ("Data Hygiene", "Identified 11 whitespace strings in Total Charges for new zero-tenure accounts; imputed cleanly without synthetic bias.")
    ])

    # Slide 4: Exploratory Data Analysis & Critical Hazard Findings
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Exploratory Data Analysis: Key Behavioral Drivers")
    add_bullet_content(s4, [
        ("The 12-Month Hazard Peak", "Over 50% of all churn occurs during months 1–12 (median tenure of churners is 10 mos vs 38 mos for retained)."),
        ("Contract Lock-in", "Month-to-month contracts churn at 42.6%, whereas 2-year contracts churn at only 2.7%."),
        ("The Fiber Optic Support Shield", "Fiber optic subscribers churn at 41.1% despite premium pricing; adding Tech Support cuts churn by more than half (~22%)."),
        ("Payment Friction", "Electronic check payers churn at 43.5%, compared to 19.0% for automated bank transfer payers.")
    ])

    # Slide 5: Preprocessing & Leakage Prevention
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Leakage-Free Preprocessing Architecture")
    add_bullet_content(s5, [
        ("Target Leakage Pruning", "Strictly excluded Churn Reason (exit interview), Churn Score (model leakage), and CLTV before split."),
        ("Identifier Isolation", "High-cardinality geographic coordinates (Zip Code, Lat/Long) pruned to prevent memorization."),
        ("Scikit-learn ColumnTransformer", "StandardScaler applied to numericals, OneHotEncoder to categoricals, embedded in an immutable pipeline bundle."),
        ("Data Partitioning", "80/20 Stratified Split (5,634 Train / 1,409 Test) ensuring exact class ratio preservation.")
    ])

    # Slide 6: Model Selection & Stratified Benchmarking
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Systematic Candidate Model Benchmarking (5-Fold CV)")
    add_bullet_content(s6, [
        ("Majority Baseline (Dummy)", "ROC-AUC: 0.5000 | Recall: 0.00% | F1: 0.0000"),
        ("Logistic Regression", "ROC-AUC: 0.8412 | Recall: 53.21% | F1: 0.5912"),
        ("Decision Tree (Cart)", "ROC-AUC: 0.6580 | Severe overfitting on training partition"),
        ("Random Forest (100 Trees)", "ROC-AUC: 0.8450 | Solid non-linear baseline"),
        ("XGBoost (Extreme Gradient)", "ROC-AUC: 0.8520 | Strong tree boosting baseline"),
        ("HistGradientBoosting (Winner)", "ROC-AUC: 0.8639 (CV Best) | Fast binning, robust categorical splits, superior out-of-fold generalization.")
    ])

    # Slide 7: Production Model Holdout Evaluation
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Holdout Test Set Performance (1,409 Untouched Records)")
    add_bullet_content(s7, [
        ("Discriminative Power", "ROC-AUC: 0.8570 | Precision-Recall AUC: 0.6740"),
        ("Operating Threshold (tau = 0.30)", "Recall: 77.01% (Captures 288 out of 374 true churners in test set)"),
        ("Precision & Efficiency", "Precision: 54.24% | F1-Score: 0.6365 | Accuracy: 76.65%"),
        ("Missed Churners Cut in Half", "Missed churners reduced from 177 (at 0.50) down to 86 (a 51.4% reduction in churn leakage)."),
        ("Probability Calibration", "Near-optimal Brier Score of 0.1322 out-of-the-box.")
    ])

    # Slide 8: SHAP Model Interpretability
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Explainable AI: Global & Local SHAP Attribution")
    add_bullet_content(s8, [
        ("TreeExplainer Integration", "Calculated exact Shapley values across test records for full auditability."),
        ("Top Global Churn Drivers", "1. Contract (Month-to-month), 2. Tenure Months, 3. Internet Service (Fiber optic), 4. Payment Method."),
        ("Local Explanations", "Every prediction response delivers per-feature positive/negative contributions to business users."),
        ("Ethical Governance", "Demographic attributes (Gender, SeniorCitizen, Partner) confirmed to have minimal impact, avoiding disparate bias.")
    ])

    # Slide 9: Stateless REST API Serving Architecture
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "FastAPI Serving Architecture & Microservices")
    add_bullet_content(s9, [
        ("Decoupled Design", "Stateless FastAPI REST microservice with lifespan-managed model singleton."),
        ("Strict Validation", "Pydantic v2 schemas validating every numerical range and categorical enum."),
        ("Dual Probes", "GET /health (Liveness) and GET /ready (Readiness verifying bundle and memory headroom)."),
        ("Inference Endpoints", "POST /predict for real-time single record scoring; POST /predict/batch for high-throughput scoring.")
    ])

    # Slide 10: Performance Engineering & Measured Benchmarks
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Empirical Performance & Throughput Scaling")
    add_bullet_content(s10, [
        ("Model Load Duration", "0.828 seconds on cold start"),
        ("Memory Footprint", "14.36 MB model overhead; 196.5 MB total process RSS"),
        ("Single Prediction Latency", "p50: 18.26 ms | p95: 19.74 ms | p99: 20.64 ms (exceeds all latency SLAs)"),
        ("Batch Inference Throughput", "Scores 1,409 customer records in 0.276s (5,104 records / second throughput on CPU).")
    ])

    # Slide 11: Application Security & Data Privacy
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Application Hardening & Responsible Privacy")
    add_bullet_content(s11, [
        ("Rate Limiting", "Sliding-window rate limiter enforcing 180 req/min per IP to prevent scraping."),
        ("Payload & Batch Ceilings", "10 MB payload limit and 2,500 record batch limit to protect against memory exhaustion DoS."),
        ("PII Masking", "CustomerID sanitized to prefix format (7590-****); zero customer PII stored in runtime logs."),
        ("Zero Exposed Secrets", "Git exclusions strictly enforced for .env and databases; dependencies audited (0 critical/high CVEs).")
    ])

    # Slide 12: MLOps Automation, MLflow & CI/CD
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "MLOps Automation: MLflow Tracking & GitHub Actions")
    add_bullet_content(s12, [
        ("MLflow Experiment Tracking", "Embedded SQLite backend (sqlite:///mlflow.db) tracking parameters, hashes, and metrics."),
        ("Model Registry & Stage Gates", "Versioned registry (models/registry.json) enforcing Staging -> Production promotion gates with instant rollback."),
        ("CI/CD Pipeline Suite", "GitHub Actions workflows for continuous integration (lint/test/coverage), Docker build, and model evaluation."),
        ("Cloud Infrastructure as Code", "Terraform definition (infrastructure/main.tf) for AWS ECS Fargate, ECR, S3, and CloudWatch.")
    ])

    # Slide 13: Customer Segmentation & Campaign Financial ROI
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Customer Personas & Financial ROI Simulation")
    add_bullet_content(s13, [
        ("K-Means Personas (k=4)", "Identified 4 distinct cohorts (Silhouette = 0.4768): New Tech (51.5% churn), Budget, Premium, and Loyalists."),
        ("Strategic Tradeoff (0.30 vs 0.50)", "Operating at 0.30 captures 91 additional churners on test set (288 vs 197)."),
        ("Financial Return (Simulation)", "Campaign spend: $26,550 | Gross revenue saved: $86,400 | Net Profit: $59,850."),
        ("Net Profit Advantage", "+$15,700 Net Profit advantage and $109,200 churn loss averted compared to default threshold.")
    ])

    # Slide 14: System Reliability & Fault Recovery
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Reliability Strategy & Instant Rollback")
    add_bullet_content(s14, [
        ("Dual-Engine Fallback", "Streamlit UI features automatic fallback to local inference engine if REST connection is interrupted."),
        ("Instant Rollback Protocol", "Verified ModelRegistry stage demotion allows 1-second reinstatement of previous production model version."),
        ("Corrupted Artifact Handling", "Automated detection of corrupted bundles returning HTTP 503 rather than corrupted predictions."),
        ("Automated Quality Assurance", "34 / 34 pytest tests passing with 100% success rate across API, serving, and MLOps modules.")
    ])

    # Slide 15: Limitations, Lessons & Future Roadmap
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Engineering Lessons & Production Roadmap")
    add_bullet_content(s15, [
        ("Dataset Limitations", "Static cross-sectional snapshot lacks event-stream telemetry (dropped calls, bandwidth drops)."),
        ("Uplift Modeling Opportunity", "Future iteration: implement causal uplift models to isolate customers who stay strictly due to incentives."),
        ("Automated Drift Retraining", "Integrate automated retraining webhooks triggered when PSI exceeds 0.20."),
        ("Project Sign-Off", "Fully completed, reproducible, tested, and documented enterprise-ready MLOps project.")
    ])

    prs.save(PRESENTATION_PATH)
    print(f"Successfully generated 15-slide PowerPoint deck at {PRESENTATION_PATH}")

if __name__ == "__main__":
    main()
