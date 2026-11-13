# Technical Debt & Known Limitations Register

This register transparently documents known limitations, engineering compromises, and recommended future enhancements for subsequent system iterations.

---

## 1. Prioritized Technical Debt Matrix

| Priority | Component | Item / Limitation | Business / Technical Impact | Remediation Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Medium** | Modeling | Synthetic Cross-Sectional Data | Static snapshot lacks event-stream telemetry (e.g. dropped call events, daily data spikes, customer support ticket velocity). | Connect stream ingestion (Kafka/Kinesis) to compute rolling temporal aggregations. |
| **Medium** | Modeling | Absence of Observed Intervention Data | Retention outreach conversion is modeled via sensitivity parameters ($25\%$ acceptance) rather than observed causal uplift. | Implement Two-Model or X-Learner causal uplift modeling to target only persuadable customers. |
| **Low** | Security | In-Memory Rate Limiting | Rate limiting counters reside in single process RAM. In multi-container deployments behind an ALB, client limits are per-pod rather than global. | Replace in-memory rate limiter with Redis-backed distributed token bucket (`redis-py`). |
| **Low** | Monitoring | Manual Retraining Trigger | While Population Stability Index (PSI) and KS tests alert on drift, retraining must currently be triggered manually. | Integrate automated GitHub Actions webhook dispatching `scripts/train.py` when PSI exceeds $0.20$. |
| **Low** | Testing | Starlette Deprecation Warning | Test output notes deprecation warning regarding `httpx` in future Starlette versions. | Upgrade test dependencies to `httpx2` once officially released. |

---

## 2. Incomplete or Deferred Capabilities
- **Remote Cloud Provisioning:** No remote AWS cloud infrastructure was deployed (preserving zero-cost user requirement); full Terraform code is ready for execution in `infrastructure/main.tf`.
- **Public Git Push:** Code has been maintained in the local Git repository without pushing to remote origins per user instructions.
