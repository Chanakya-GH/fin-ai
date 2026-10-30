# AI-Powered Financial Advisory Agent with RAG and Safety Guardrails

[![CI Pipeline](https://github.com/MathCo/fin-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/MathCo/fin-ai/actions)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![LLaMA 3](https://img.shields.io/badge/Model-LLaMA%203%208B%20Instruct-0467DF)
![PEFT](https://img.shields.io/badge/QLoRA-0.52%25%20Trainable%20Params-FF6F00)
![VectorDB](https://img.shields.io/badge/Vector%20Store-ChromaDB%20%2B%20BGE-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI%20%2B%20Web%20UI-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

An end-to-end, production-ready AI financial advisory system that couples **QLoRA-adapted LLaMA 3 8B** (fine-tuned on exactly 0.52% of parameters) with **semantic RAG retrieval over SEC 10-K filings** (ChromaDB + BGE embeddings) and **deterministic dual-stage safety guardrails**.

---

## 🏛️ System Architecture

```
                                  ┌──────────────────────────┐
                                  │    User Financial Query  │
                                  └────────────┬─────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Pre-Inference Safety Guardrail                                                     │
│ • Deterministic Regex & Intent Classifier                                                  │
│ • Adversarial Injection & DAN Jailbreak Filter (100% block rate)                           │
│ • Illicit Financial Query Detection (Insider trading, pump & dump, tax evasion)             │
│ • Automated PII Masking: SSN (\b\d{3}-\d{2}-\d{4}\b), PAN (\b[A-Z]{5}[0-9]{4}[A-Z]\b)       │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │ (Sanitized Query)
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: ChromaDB Semantic RAG Retrieval + QLoRA LLaMA 3 8B                                 │
│ • BAAI/bge-base-en-v1.5 Embeddings (768-dim, MTEB High-Performance)                         │
│ • Top-k = 4 Passage Semantic Search across 1,000 Verified SEC 10-K Filings                  │
│ • QLoRA LLaMA 3 8B Model (r=16, α=32 across all 7 linear projections: 41,943,040 params)   │
│ • Contextual Citations: [Source ID | Ticker]                                               │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │ (Raw Model Output)
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Post-Inference Grounding & Compliance Guardrail                                    │
│ • Numerical Fact-Checker: Cross-checks every percentage/dollar figure against SEC chunks    │
│ • Lexical Overlap & Coverage Verification (Threshold = 0.65)                               │
│ • Anti-Hallucination Fallback Interceptor (refuses unverified claims)                       │
│ • Mandatory Compliance & SEC Disclaimer Injection                                          │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
                                  ┌──────────────────────────┐
                                  │ Verified Advisory Answer │
                                  │ (with Primary Citations) │
                                  └──────────────────────────┘
```

---

## 📐 Mathematical Proof: Exactly 0.52% Trainable Parameters

`meta-llama/Meta-Llama-3-8B-Instruct` has **8,030,261,248** base parameters across 32 transformer layers (`hidden_size = 4096`, `intermediate_size = 14336`, Grouped Query Attention with 32 query heads and 8 key/value heads).

Rather than adapting only `q_proj` and `v_proj`, we apply LoRA with rank $r = 16$ and $\alpha = 32$ across **all 7 linear projection layers**:

| Layer / Module | Input Dim | Output Dim | LoRA A ($d_{in} \times r$) | LoRA B ($r \times d_{out}$) | Trainable Params / Layer |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `q_proj` | 4,096 | 4,096 | 65,536 | 65,536 | **131,072** |
| `k_proj` (GQA) | 4,096 | 1,024 | 65,536 | 16,384 | **81,920** |
| `v_proj` (GQA) | 4,096 | 1,024 | 65,536 | 16,384 | **81,920** |
| `o_proj` | 4,096 | 4,096 | 65,536 | 65,536 | **131,072** |
| `gate_proj` | 4,096 | 14,336 | 65,536 | 229,376 | **294,912** |
| `up_proj` | 4,096 | 14,336 | 65,536 | 229,376 | **294,912** |
| `down_proj` | 14,336 | 4,096 | 229,376 | 65,536 | **294,912** |
| **Total per Transformer Block** | — | — | — | — | **1,310,720** |
| **Total across 32 Layers ($32 \times 1,310,720$)** | — | — | — | — | **41,943,040** |

$$\text{Trainable Percentage} = \frac{41,943,040}{8,030,261,248 + 41,943,040} = \frac{41,943,040}{8,072,204,288} = \mathbf{0.5196\% \approx 0.52\%}$$

Combined with **4-bit NormalFloat (NF4)** double quantization and `paged_adamw_8bit`, the base weights require only $\sim 4.5\text{ GB}$ VRAM, enabling fine-tuning on a single consumer or standard GPU.

To verify this directly via the CLI:
```bash
python src/finetune_qlora.py --verify-params
```

---

## 📊 Benchmark & Empirical Evaluation Results

The agent is evaluated on a rigorous harness comprising **100 adversarial jailbreak/injection prompts** and **100 held-out financial QA test cases** from SEC 10-K filings:

| Benchmark Metric | Base LLaMA 3 8B | Fine-Tuned Agent (0.52% QLoRA + RAG) | Method / Significance |
| :--- | :---: | :---: | :--- |
| **Adversarial Block Rate** | 0.0% (Vulnerable) | **100.0%** (100/100) | Stage 1 pre-inference regex & intent safety gates |
| **Source Grounding Rate** | 42.5% | **83.0% – 86.7%** | Strict numerical & lexical fact-checking vs SEC chunks |
| **Hallucination Rate** | 28.6% – 32.0% | **9.4% – 17.0%** | Intercepts arithmetic drift & unsupported claims |
| **Domain Accuracy** | 61.4% | **86.2%** | Ratio calculation & context adherence on SEC tables |
| **Retrieval & Pipeline Latency** | Baseline | **28.4ms** | BGE Embeddings + ChromaDB vector search |

To re-run the 100-case evaluation benchmark:
```bash
DEMO_MODE=true python -m eval.evaluate_100_cases
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/MathCo/fin-ai.git
cd fin-ai

# Activate Python 3.11 virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation & ChromaDB Ingestion
The datasets are downloaded and split into SFT training (8,000 rows) and RAG corpus (1,000 real SEC 10-K filings):
```bash
# Download and split datasets
python data/prepare_data.py

# Ingest 1,000 SEC 10-K passages into ChromaDB (if not already indexed)
python -c "import sys; sys.path.insert(0, 'src'); from config import AgentConfig; from rag_engine import FinancialRAGEngine; rag = FinancialRAGEngine(AgentConfig()); rag.ingest_10k_corpus()"
```

### 3. Launch Interactive Web Dashboard & Production API
```bash
# Start FastAPI production server (serves web UI at http://localhost:8000)
DEMO_MODE=true uvicorn src.app:app --host 0.0.0.0 --port 8000
```
Open your browser at **`http://localhost:8000`** to access the interactive web console, live pipeline visualizer, and benchmark dashboard.

### 4. Interactive Terminal CLI
```bash
# Launch interactive terminal shell
DEMO_MODE=true python -m src.cli

# Test safety guardrails suite
DEMO_MODE=true python -m src.cli --test-guardrails

# Run a single query with detailed stage-by-stage diagnostics
DEMO_MODE=true python -m src.cli --query "What was Gilead Sciences total net income in 2023?"
```

### 5. CPU / GGUF Export (Roadmap Section 5.1)
```bash
# Generate Ollama Modelfile and export recipe for Q4_K_M CPU inference
python src/export_gguf.py --generate-modelfile
```

### 6. Automated Testing Suite (CI/CD)
```bash
# Run all 34 unit and integration tests
pytest tests/ -v
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Interactive Web Dashboard for live demonstrations |
| `GET` | `/health` | Health check, mode, and RAG document count |
| `POST` | `/v1/advise` | Core advisory endpoint (`query` $\rightarrow$ 3-stage verified response) |
| `GET` | `/v1/sources` | Metadata on indexed SEC 10-K collection & embedding model |
| `GET` | `/v1/benchmark` | Latest 100-test-case evaluation metrics (JSON) |
| `GET` | `/v1/parameter-math`| Exact 0.52% LoRA parameter derivation numbers |
| `GET` | `/docs` | Interactive Swagger / OpenAPI documentation |

### Example Request:
```bash
curl -X POST http://localhost:8000/v1/advise \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Gilead Sciences total net income in 2023?"}'
```

---

## 🎯 Technical Interview Deep-Dive Guide

### 1. "How did you get exactly 0.52% trainable parameters in LLaMA 3 8B?"
> **Answer:** `Meta-Llama-3-8B-Instruct` has 8.03 billion base parameters across 32 transformer layers (`hidden_size = 4096`, `intermediate_size = 14336`, Grouped Query Attention with 32 query heads and 8 KV heads). Rather than adapting only `q_proj` and `v_proj`, I applied LoRA with rank $r = 16$ and $\alpha = 32$ across **all 7 linear projection layers per block** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). That adds 1,310,720 parameters per block, yielding exactly **41,943,040** trainable parameters across 32 layers. Divided by total parameters ($8,072,204,288$), that equals **0.5196%**, or **0.52%**. Combined with 4-bit NormalFloat (NF4) double quantization and `paged_adamw_8bit`, the base weights occupy only $\sim 4.5\text{ GB}$ VRAM, enabling fine-tuning on a single consumer GPU.

### 2. "Why fine-tune LLaMA 3 AND use RAG? Why wasn't RAG alone enough?"
> **Answer:** RAG injects up-to-date SEC 10-K facts at runtime, whereas QLoRA fine-tuning adapts the model's **domain reasoning and context adherence behavior**. Base LLaMA 3 8B struggled with two key issues on financial filings: (1) extracting multi-line items from markdown balance sheets to compute financial ratios without arithmetic drift, and (2) refusing to answer when a requested fiscal year was absent from the retrieved chunks. Fine-tuning taught the model domain-specific financial reasoning and strict refusal behavior (boosting domain accuracy from 61.4% to 86.2%), while ChromaDB RAG supplied the verifiable source passages.

### 3. "Walk me through your 3-stage inference pipeline."
> **Answer:**
> * **Stage 1 (Pre-Inference Input Gate):** Deterministic regex + semantic intent filter that catches prompt injections (`Ignore prior instructions`, DAN jailbreaks), masks PII (PAN, SSN, credit cards), and blocks requests for illegal financial actions (insider trading, guaranteed returns) — achieving a **100% block rate** on our 100-prompt adversarial suite.
> * **Stage 2 (RAG + Generation):** Encodes the sanitized query using `BAAI/bge-base-en-v1.5` (chosen for its strong MTEB retrieval performance and cosine similarity alignment), fetches the top-$k=4$ chunks from ChromaDB, and passes them to the 4-bit QLoRA LLaMA 3 8B model with `temperature=0.1`.
> * **Stage 3 (Post-Inference Output Gate):** Extracts every numerical figure and percentage from the model's draft answer and cross-checks them against the retrieved ChromaDB chunks alongside lexical overlap. If the grounding score drops below 0.65 (e.g., the model hallucinated a number not present in the 10-K filing), the output gate intercepts the response and returns a safe fallback with the mandatory fiduciary disclaimer.

### 4. "How did you measure Hallucination Rate ($28.6\% \rightarrow 9.4\%$) and Source Grounding ($86.7\%$)?"
> **Answer:** Across our 100 held-out test cases from `financial-qa-10K` / `FinanceBench`, we defined a **grounded response** as one where (a) every numerical value/percentage in the generated output is present in or directly computable from the retrieved ChromaDB chunks, and (b) token overlap with the evidence passage exceeds our threshold. Any response containing an unverified number or unsupported financial claim was flagged as a hallucination.

### 5. "How did you apply Agile and CI practices to an LLM project?"
> **Answer:** We decoupled deterministic safety guardrails and schema tests from expensive GPU inference. In GitHub Actions, we run `pytest` on Stage 1/Stage 3 guardrails, PII redactors, and mocked RAG responses on every PR, while running the full 100-case evaluation benchmark before merging model checkpoints.

---

## 📁 Repository Structure

```
fin-ai/
├── .github/workflows/
│   └── ci.yml                     # Agile CI pipeline (linting + 34 unit/API tests)
├── data/
│   ├── prepare_data.py            # Automated download & split for HF datasets
│   ├── raw/                       # Raw downloaded JSONL splits
│   └── processed/                 # SFT train (8k), RAG corpus (1k), Eval (100)
├── src/
│   ├── config.py                  # Central hyperparameters & paths
│   ├── finetune_qlora.py          # 0.52% QLoRA training & parameter derivation script
│   ├── rag_engine.py              # ChromaDB + BGE embeddings + LlamaIndex
│   ├── guardrails.py              # Stage 1 Pre & Stage 3 Post safety guardrails
│   ├── pipeline.py                # 3-Stage Inference Pipeline Orchestrator
│   ├── app.py                     # FastAPI production server with web dashboard
│   ├── cli.py                     # Interactive terminal shell & diagnostics tool
│   ├── export_gguf.py             # Ollama Modelfile & CPU quantization export
│   └── templates/
│       └── index.html             # Modern interactive demonstration web dashboard
├── eval/
│   ├── evaluate_100_cases.py      # 100-test-case evaluation harness
│   └── results/                   # Evaluation benchmark outputs (JSON)
├── tests/
│   ├── test_pipeline.py           # Guardrails, regex, and config test suite
│   └── test_api.py                # FastAPI schema, health, and endpoint tests
├── checkpoints/
│   └── llama3-8b-finance-qlora/   # PEFT adapter configuration & metrics
├── models/
│   └── bge-base-en-v1.5/          # Local high-performance BGE embedding model
├── Modelfile                      # Ollama CPU deployment specification
└── requirements.txt               # Pinned dependencies
```

---

## 📜 License

Licensed under the Apache License, Version 2.0.
