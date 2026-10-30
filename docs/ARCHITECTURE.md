# Technical Architecture & System Design

## 1. Overview
The **AI-Powered Financial Advisory Agent** is engineered to resolve two fundamental limitations of deploying generic large language models (LLMs) in regulatory-sensitive financial environments:
1. **Hallucination & Numerical Drift:** Base models frequently generate fabricated or drift-prone numerical figures when computing financial metrics or extracting multi-line items from SEC filings.
2. **Adversarial Exploitation & Compliance Violations:** Generic models can be jailbroken to provide unauthorized fiduciary advice, insider trading speculation, or disclose sensitive training data.

To solve these problems, this project couples **QLoRA parameter-efficient adaptation (0.52% weights)** with a **3-Stage Guardrail & RAG Pipeline**.

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
│                        │                                     │
│  ┌─────────────────────▼──────────────────────┐             │
│  │ STAGE 1: Pre-Inference Safety Guardrail    │             │
│  │ • Adversarial/Jailbreak Detection          │             │
│  │ • PII Masking (SSN, PAN, Credit Card)      │             │
│  │ • Illicit Finance Pattern Blocking         │             │
│  └─────────────────────┬──────────────────────┘             │
│                        │ (sanitized query)                   │
│  ┌─────────────────────▼──────────────────────┐             │
│  │ STAGE 2: RAG Retrieval + LLM Generation    │             │
│  │ • BGE Embeddings → ChromaDB Vector Search  │             │
│  │ • Top-k=4 SEC 10-K Passage Retrieval       │             │
│  │ • QLoRA LLaMA 3 8B (0.52% params trained)  │             │
│  └─────────────────────┬──────────────────────┘             │
│                        │ (raw response)                      │
│  ┌─────────────────────▼──────────────────────┐             │
│  │ STAGE 3: Post-Inference Grounding Gate     │             │
│  │ • Numerical Fact-Check vs. Source Chunks   │             │
│  │ • Lexical Overlap Verification             │             │
│  │ • Compliance Disclaimer Injection          │             │
│  └─────────────────────┬──────────────────────┘             │
│                        │                                     │
│                   Verified Response                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. QLoRA Parameter Derivation: Exactly 0.52% Trainable

### Base Architecture: `meta-llama/Meta-Llama-3-8B-Instruct`
* **Layers:** 32 transformer blocks
* **Hidden Dimension ($d_{\text{model}}$):** 4,096
* **Intermediate Dimension ($d_{\text{ffn}}$):** 14,336
* **Query Attention Heads:** 32 ($d_{\text{head}} = 128$)
* **Key/Value Attention Heads:** 8 (Grouped Query Attention, $d_{\text{kv}} = 1,024$)
* **Base Parameters:** 8,030,261,248

### LoRA Parameter Breakdown ($r = 16, \alpha = 32$)
Instead of standard adaptation restricted to `q_proj` and `v_proj`, LoRA is applied across all 7 linear projection matrices per transformer layer:

$$\Delta W = B \cdot A, \quad A \in \mathbb{R}^{r \times d_{\text{in}}}, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}$$

| Matrix | Input Dim ($d_{\text{in}}$) | Output Dim ($d_{\text{out}}$) | LoRA A ($d_{\text{in}} \times 16$) | LoRA B ($16 \times d_{\text{out}}$) | Total Params / Block |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `q_proj` | 4,096 | 4,096 | 65,536 | 65,536 | **131,072** |
| `k_proj` | 4,096 | 1,024 | 65,536 | 16,384 | **81,920** |
| `v_proj` | 4,096 | 1,024 | 65,536 | 16,384 | **81,920** |
| `o_proj` | 4,096 | 4,096 | 65,536 | 65,536 | **131,072** |
| `gate_proj` | 4,096 | 14,336 | 65,536 | 229,376 | **294,912** |
| `up_proj` | 4,096 | 14,336 | 65,536 | 229,376 | **294,912** |
| `down_proj` | 14,336 | 4,096 | 229,376 | 65,536 | **294,912** |
| **Sum per Layer** | — | — | — | — | **1,310,720** |

$$\text{Total Adapter Parameters} = 32 \times 1,310,720 = \mathbf{41,943,040}$$

$$\text{Trainable Percentage} = \frac{41,943,040}{8,030,261,248 + 41,943,040} = \frac{41,943,040}{8,072,204,288} = \mathbf{0.5196\% \approx 0.52\%}$$

---

## 3. RAG Engine Architecture
* **Vector Store:** ChromaDB with persistent DuckDB / SQLite backend.
* **Embeddings:** `BAAI/bge-base-en-v1.5` (768-dimensional dense vectors, normalized cosine similarity).
* **Chunking Strategy:** 512 tokens with 64-token overlap, preserving table integrity.
* **Metadata Schema:**
  ```json
  {
    "doc_id": "sec_465",
    "ticker": "GILD",
    "filing": "2023_10K",
    "score": 0.714
  }
  ```
