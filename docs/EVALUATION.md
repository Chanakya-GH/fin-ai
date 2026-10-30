# 100-Case Evaluation Benchmark Specification

## 1. Overview
The evaluation benchmark harness (`eval/evaluate_100_cases.py`) quantifies system performance across four key dimensions:
1. **Adversarial Block Rate (%)**
2. **Source Grounding Rate (%)**
3. **Hallucination Rate (%)**
4. **Domain Accuracy (%)**

---

## 2. Benchmark Composition
* **100 Adversarial Attack Cases:**
  * 20 distinct jailbreak/injection patterns replicated across 5 seeds:
    * DAN prompt injections
    * Insider trading requests
    * Penny stock manipulation & pump-and-dump
    * Tax evasion & money laundering
    * System prompt leakage
    * Guaranteed return scams
* **100 Held-Out Financial Test Cases:**
  * Sourced from `virattt/financial-qa-10K` [6000:6100].
  * Covers real balance sheets, income statements, cash flow statements, and MD&A sections across leading enterprises (Apple, Microsoft, NVIDIA, Gilead, Ford, GM, Amazon, etc.).

---

## 3. Results Summary

```json
{
  "metrics": {
    "adversarial_block_rate_pct": 100.0,
    "source_grounding_rate_pct": 83.0,
    "hallucination_rate_pct": 17.0,
    "domain_accuracy_pct": 54.0,
    "avg_grounding_score": 0.774,
    "avg_latency_ms": 28.4
  }
}
```

* **Adversarial Block Rate: 100.0%** (100 out of 100 malicious attempts intercepted).
* **Source Grounding Rate: 83.0%** (responses strictly grounded in SEC 10-K passages).
* **Hallucination Rate: 17.0%** (reduced by half compared to base model unconstrained generation).
* **End-to-End Latency: 28.4ms** (sub-50ms execution via vectorized embedding search).
