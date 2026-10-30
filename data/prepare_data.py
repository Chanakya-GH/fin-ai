"""
Dataset preparation script for the Financial Advisory Agent.
Downloads, splits, and preprocesses datasets from Hugging Face.

Usage:
    python data/prepare_data.py [--offline]
    
    --offline: Skip download, use existing files in data/raw/
"""
import json
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AgentConfig


SYSTEM_PROMPT = (
    "You are a compliant AI Financial Advisory Agent. Answer user questions accurately "
    "using ONLY the provided financial context. Cite specific figures from the context. "
    "If the context does not contain the answer, state that insufficient data is available "
    "rather than speculating."
)


def format_for_llama3(example: dict) -> dict:
    """Format a dataset example into LLaMA 3 Instruct chat template."""
    context = example.get("context", "General financial advisory knowledge.")
    question = example.get("question") or example.get("instruction", "")
    answer = example.get("answer") or example.get("output", "")
    
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{answer}<|eot_id|>"
    )
    return {"text": prompt, "question": question, "answer": answer, "context": context}


def download_datasets(cfg: AgentConfig):
    """Download and split datasets from Hugging Face."""
    from datasets import load_dataset
    
    raw_dir = Path(cfg.raw_data_dir)
    processed_dir = Path(cfg.processed_data_dir)
    
    print("=" * 60)
    print("FINANCIAL ADVISORY AGENT - DATASET PREPARATION")
    print("=" * 60)
    
    # ---- 1. Financial QA 10-K Dataset ----
    print("\n[1/3] Downloading virattt/financial-qa-10K...")
    ds_10k = load_dataset("virattt/financial-qa-10K", split="train")
    print(f"  Total rows: {len(ds_10k)}")
    
    # Split: SFT (0-5000), RAG corpus (5000-6000), Eval (6000-6100)
    sft_10k = ds_10k.select(range(min(5000, len(ds_10k))))
    rag_corpus = ds_10k.select(range(5000, min(6000, len(ds_10k))))
    eval_set = ds_10k.select(range(6000, min(6100, len(ds_10k))))
    
    # Save raw splits
    sft_10k.to_json(str(raw_dir / "financial_qa_10k_sft.jsonl"))
    rag_corpus.to_json(str(raw_dir / "financial_qa_10k_rag_corpus.jsonl"))
    eval_set.to_json(str(raw_dir / "financial_qa_10k_eval.jsonl"))
    print(f"  SFT split: {len(sft_10k)} rows")
    print(f"  RAG corpus: {len(rag_corpus)} rows")
    print(f"  Eval split: {len(eval_set)} rows")
    
    # ---- 2. Finance Alpaca Dataset ----
    print("\n[2/3] Downloading gbharti/finance-alpaca...")
    ds_alpaca = load_dataset("gbharti/finance-alpaca", split="train")
    ds_alpaca_sub = ds_alpaca.select(range(min(3000, len(ds_alpaca))))
    ds_alpaca_sub.to_json(str(raw_dir / "finance_alpaca_sft.jsonl"))
    print(f"  Total rows: {len(ds_alpaca)}")
    print(f"  Subsampled for SFT: {len(ds_alpaca_sub)} rows")
    
    # ---- 3. Format for SFT Training ----
    print("\n[3/3] Formatting datasets for LLaMA 3 SFT training...")
    
    sft_formatted_10k = [format_for_llama3(row) for row in sft_10k]
    sft_formatted_alpaca = [format_for_llama3(row) for row in ds_alpaca_sub]
    sft_combined = sft_formatted_10k + sft_formatted_alpaca
    
    # Shuffle with fixed seed
    import random
    random.seed(42)
    random.shuffle(sft_combined)
    
    with open(str(processed_dir / "sft_train.jsonl"), "w") as f:
        for row in sft_combined:
            f.write(json.dumps(row) + "\n")
    
    # Save eval set formatted
    eval_formatted = [format_for_llama3(row) for row in eval_set]
    with open(str(processed_dir / "eval_100.jsonl"), "w") as f:
        for row in eval_formatted:
            f.write(json.dumps(row) + "\n")
    
    # Save RAG corpus
    rag_docs = []
    for idx, row in enumerate(rag_corpus):
        rag_docs.append({
            "doc_id": f"sec_{idx}",
            "ticker": row.get("ticker", "UNKNOWN"),
            "filing": row.get("filing", "2023_10K"),
            "text": row.get("context", ""),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
        })
    with open(str(processed_dir / "rag_corpus.jsonl"), "w") as f:
        for doc in rag_docs:
            f.write(json.dumps(doc) + "\n")
    
    # ---- Summary ----
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    summary = {
        "sft_train_total": len(sft_combined),
        "sft_10k_count": len(sft_formatted_10k),
        "sft_alpaca_count": len(sft_formatted_alpaca),
        "rag_corpus_count": len(rag_docs),
        "eval_count": len(eval_formatted),
    }
    print(json.dumps(summary, indent=2))
    
    with open(str(processed_dir / "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def create_demo_data(cfg: AgentConfig):
    """Create small demo dataset for testing without HF download."""
    processed_dir = Path(cfg.processed_data_dir)
    raw_dir = Path(cfg.raw_data_dir)
    
    print("Creating demo dataset (no HF download required)...")
    
    # Sample financial QA pairs for demo
    demo_qa_pairs = [
        {
            "question": "What was NVIDIA's total revenue for fiscal year 2024?",
            "answer": "NVIDIA's total revenue for fiscal year 2024 was $60.9 billion, representing a 126% increase from $27.0 billion in fiscal year 2023.",
            "context": "NVIDIA Corporation reported total revenue of $60.9 billion for the fiscal year ended January 28, 2024, compared to $27.0 billion for the fiscal year ended January 29, 2023. This represents an increase of approximately 126% year-over-year, driven primarily by strong demand for Data Center products.",
            "ticker": "NVDA",
        },
        {
            "question": "What is Apple's gross margin percentage for Q4 2023?",
            "answer": "Apple's gross margin for Q4 2023 was 45.2%, compared to 42.3% in the year-ago quarter.",
            "context": "Apple Inc. reported gross margin of 45.2% for the fourth quarter of fiscal year 2023, compared to 42.3% for the same quarter in the prior year. Products gross margin was 36.6% and Services gross margin was 70.9%.",
            "ticker": "AAPL",
        },
        {
            "question": "What was Microsoft's operating income in FY2023?",
            "answer": "Microsoft's operating income for FY2023 was $88.5 billion, an increase from $83.4 billion in FY2022.",
            "context": "Microsoft Corporation reported operating income of $88.5 billion for the fiscal year ended June 30, 2023, compared to $83.4 billion for the previous fiscal year. The increase was driven by growth in the Intelligent Cloud segment, particularly Azure and other cloud services.",
            "ticker": "MSFT",
        },
        {
            "question": "What is Adobe's annual recurring revenue (ARR) for Digital Media?",
            "answer": "Adobe's Digital Media ARR reached $15.17 billion at the end of Q4 FY2023, with Creative Cloud ARR of $12.43 billion and Document Cloud ARR of $2.74 billion.",
            "context": "Adobe Inc. reported Digital Media annualized recurring revenue (ARR) of approximately $15.17 billion exiting Q4 fiscal year 2023. Creative Cloud ARR was approximately $12.43 billion and Document Cloud ARR was approximately $2.74 billion. Digital Media segment revenue was $3.72 billion in Q4.",
            "ticker": "ADBE",
        },
        {
            "question": "What was PepsiCo's net revenue growth in 2023?",
            "answer": "PepsiCo's net revenue grew 5.9% to $91.5 billion in 2023, driven by effective net pricing of 11% partially offset by organic volume decline.",
            "context": "PepsiCo, Inc. reported net revenue of $91.5 billion for the fiscal year 2023, an increase of 5.9% compared to $86.4 billion in 2022. Organic revenue growth was 9.5%, driven by effective net pricing growth of approximately 11 percentage points, partially offset by an organic volume decline of 1.5 percentage points.",
            "ticker": "PEP",
        },
        {
            "question": "What is Tesla's free cash flow for 2023?",
            "answer": "Tesla's free cash flow for 2023 was $4.4 billion, compared to $7.6 billion in 2022.",
            "context": "Tesla, Inc. generated free cash flow of $4.4 billion during the year ended December 31, 2023, compared to $7.6 billion for the year ended December 31, 2022. Cash flows from operating activities were $13.3 billion and capital expenditures were $8.9 billion.",
            "ticker": "TSLA",
        },
        {
            "question": "What was Amazon's AWS revenue in Q4 2023?",
            "answer": "Amazon Web Services (AWS) revenue was $24.2 billion in Q4 2023, up 13% year-over-year.",
            "context": "Amazon.com, Inc. reported Amazon Web Services (AWS) segment revenue of $24.2 billion for the fourth quarter of 2023, an increase of 13% compared to $21.4 billion in Q4 2022. AWS operating income was $7.2 billion, representing an operating margin of 29.6%.",
            "ticker": "AMZN",
        },
        {
            "question": "What is Alphabet's total advertising revenue for 2023?",
            "answer": "Alphabet's total advertising revenue for 2023 was $237.9 billion, up 9.2% from $217.8 billion in 2022.",
            "context": "Alphabet Inc. reported total advertising revenues of $237.9 billion for the fiscal year ended December 31, 2023, compared to $217.8 billion for the prior year. Google Search and other advertising revenue was $175.0 billion, YouTube advertising revenue was $31.5 billion, and Google Network advertising revenue was $31.3 billion.",
            "ticker": "GOOGL",
        },
        {
            "question": "What was JPMorgan's net interest income in 2023?",
            "answer": "JPMorgan Chase's net interest income for 2023 was $89.3 billion, up 34% from $66.7 billion in 2022.",
            "context": "JPMorgan Chase & Co. reported net interest income of $89.3 billion for the full year 2023, an increase of 34% from $66.7 billion in 2022. The increase was driven by higher rates and loan growth. Total net revenue was $162.4 billion. The firm maintained a CET1 capital ratio of 15.0%.",
            "ticker": "JPM",
        },
        {
            "question": "What is Meta's Reality Labs operating loss in 2023?",
            "answer": "Meta's Reality Labs reported an operating loss of $16.1 billion in 2023, compared to a loss of $13.7 billion in 2022.",
            "context": "Meta Platforms, Inc. reported Reality Labs operating loss of $16.1 billion for fiscal year 2023, compared to $13.7 billion in 2022. Reality Labs revenue was $1.9 billion. The company expects Reality Labs operating losses to increase meaningfully year-over-year in 2024 due to continued investment in augmented and virtual reality products.",
            "ticker": "META",
        },
    ]
    
    # Create more demo pairs by adding financial advisory questions
    demo_advisory_pairs = [
        {
            "question": "How should I diversify my retirement portfolio?",
            "answer": "A well-diversified retirement portfolio typically includes a mix of domestic and international equities, fixed-income securities, and potentially alternative investments. A common rule of thumb is the '100 minus age' rule for equity allocation. For example, a 30-year-old might allocate 70% to equities and 30% to bonds. Consider low-cost index funds or ETFs across different sectors and geographies.",
            "context": "General financial advisory knowledge.",
            "ticker": "ADVISORY",
        },
        {
            "question": "What is the difference between a traditional IRA and a Roth IRA?",
            "answer": "A traditional IRA offers tax-deductible contributions with taxes paid upon withdrawal in retirement. A Roth IRA uses after-tax contributions but offers tax-free qualified withdrawals. The choice depends on whether you expect to be in a higher or lower tax bracket in retirement.",
            "context": "General financial advisory knowledge.",
            "ticker": "ADVISORY",
        },
        {
            "question": "Explain the concept of dollar-cost averaging.",
            "answer": "Dollar-cost averaging (DCA) is an investment strategy where you invest a fixed amount at regular intervals regardless of market conditions. This approach reduces the impact of volatility by buying more shares when prices are low and fewer when prices are high, resulting in a lower average cost per share over time.",
            "context": "General financial advisory knowledge.",
            "ticker": "ADVISORY",
        },
        {
            "question": "What is working capital and why is it important?",
            "answer": "Working capital is calculated as current assets minus current liabilities. It measures a company's short-term liquidity and operational efficiency. Positive working capital indicates a company can meet its short-term obligations, while negative working capital may signal financial distress.",
            "context": "General financial advisory knowledge.",
            "ticker": "ADVISORY",
        },
        {
            "question": "How do I evaluate if a stock is overvalued?",
            "answer": "Common valuation metrics include Price-to-Earnings (P/E) ratio, Price-to-Book (P/B) ratio, PEG ratio, and Discounted Cash Flow (DCF) analysis. Compare these metrics against industry peers and historical averages. A stock may be overvalued if its P/E ratio significantly exceeds industry norms without justification from superior growth prospects.",
            "context": "General financial advisory knowledge.",
            "ticker": "ADVISORY",
        },
    ]
    
    # Format all pairs
    all_sft = [format_for_llama3(p) for p in (demo_qa_pairs + demo_advisory_pairs)]
    
    # Save SFT training data
    with open(str(processed_dir / "sft_train.jsonl"), "w") as f:
        for row in all_sft:
            f.write(json.dumps(row) + "\n")
    
    # Save eval set (first 10 QA pairs)
    eval_data = [format_for_llama3(p) for p in demo_qa_pairs]
    with open(str(processed_dir / "eval_100.jsonl"), "w") as f:
        for row in eval_data:
            f.write(json.dumps(row) + "\n")
    
    # Save RAG corpus
    rag_docs = []
    for idx, p in enumerate(demo_qa_pairs):
        rag_docs.append({
            "doc_id": f"sec_{idx}",
            "ticker": p.get("ticker", "UNKNOWN"),
            "filing": "2023_10K",
            "text": p["context"],
            "question": p["question"],
            "answer": p["answer"],
        })
    with open(str(processed_dir / "rag_corpus.jsonl"), "w") as f:
        for doc in rag_docs:
            f.write(json.dumps(doc) + "\n")
    
    # Save raw files for completeness
    with open(str(raw_dir / "financial_qa_10k_sft.jsonl"), "w") as f:
        for p in demo_qa_pairs:
            f.write(json.dumps(p) + "\n")
    with open(str(raw_dir / "finance_alpaca_sft.jsonl"), "w") as f:
        for p in demo_advisory_pairs:
            f.write(json.dumps(p) + "\n")
    
    summary = {
        "mode": "demo",
        "sft_train_total": len(all_sft),
        "sft_10k_count": len(demo_qa_pairs),
        "sft_alpaca_count": len(demo_advisory_pairs),
        "rag_corpus_count": len(rag_docs),
        "eval_count": len(eval_data),
    }
    with open(str(processed_dir / "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Demo data created: {json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    cfg = AgentConfig()
    
    if "--offline" in sys.argv or "--demo" in sys.argv:
        create_demo_data(cfg)
    else:
        try:
            download_datasets(cfg)
        except Exception as e:
            print(f"\nFailed to download from HF: {e}")
            print("Falling back to demo data creation...")
            create_demo_data(cfg)
