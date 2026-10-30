"""
3-Stage Inference Pipeline Orchestrator.

Orchestrates the complete financial advisory pipeline:
  Stage 1: Pre-Inference Safety Guardrail (adversarial detection, PII masking)
  Stage 2: ChromaDB RAG Retrieval + Fine-Tuned LLaMA 3 8B Inference  
  Stage 3: Post-Inference Grounding & Compliance Guardrail

Supports two modes:
  - Full mode (load_llm=True): Uses the actual QLoRA fine-tuned model
  - Demo mode (load_llm=False): Uses intelligent mock responses for demonstration
"""
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import AgentConfig
from rag_engine import FinancialRAGEngine
from guardrails import FinancialSafetyGuardrails


class FinancialAdvisoryPipeline:
    """
    End-to-end financial advisory inference pipeline.
    
    Combines pre-inference guardrails, RAG retrieval, LLM generation,
    and post-inference grounding verification into a single pipeline.
    """
    
    def __init__(self, cfg: AgentConfig, load_llm: bool = True):
        self.cfg = cfg
        self.guardrails = FinancialSafetyGuardrails()
        self.rag = FinancialRAGEngine(cfg)
        self.model = None
        self.tokenizer = None
        self._demo_mode = not load_llm or cfg.demo_mode

        if load_llm and not cfg.demo_mode:
            self._init_model()
        else:
            print("[DEMO MODE] Running without LLM - using intelligent mock responses")

    def _init_model(self):
        """Initialize the 4-bit quantized LLaMA 3 + QLoRA adapter."""
        print(f"Loading base model: {self.cfg.base_model_id}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        adapter_path = Path(self.cfg.adapter_dir)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            print(f"Loading QLoRA adapter from: {self.cfg.adapter_dir}")
            self.model = PeftModel.from_pretrained(base_model, self.cfg.adapter_dir)
        else:
            print(f"WARNING: No adapter found at {self.cfg.adapter_dir}. Using base model.")
            self.model = base_model
        
        self.model.eval()
        print("Model loaded successfully.")

    def generate_llm_answer(self, query: str, chunks: list[dict]) -> str:
        """
        Generate an answer using the fine-tuned LLM with retrieved context.
        
        Args:
            query: Sanitized user query.
            chunks: Retrieved RAG context chunks.
            
        Returns:
            Generated response string.
        """
        if self._demo_mode:
            return self._generate_demo_answer(query, chunks)
        
        context_block = "\n\n".join(
            f"[Source {c['doc_id']} | {c['ticker']}]: {c['text']}" for c in chunks
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a compliant AI Financial Advisory Agent. Use ONLY the provided "
                    "sources to answer the user's question and cite the Source ID."
                ),
            },
            {
                "role": "user",
                "content": f"Sources:\n{context_block}\n\nQuestion: {query}",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _generate_demo_answer(self, query: str, chunks: list[dict]) -> str:
        """
        Generate an intelligent grounded answer using retrieved SEC 10-K context chunks.
        
        Simulates what the fine-tuned LLM produces by answering using only the 
        verified context and citing the source document.
        """
        if not chunks:
            return "I do not have sufficient data in my knowledge base to answer this question accurately."
        
        best_chunk = chunks[0]
        context_text = best_chunk["text"].strip()
        ticker = best_chunk.get("ticker", "SEC")
        doc_id = best_chunk.get("doc_id", "N/A")
        
        # Pick sentences matching query terms
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        sentences = [s.strip() for s in context_text.split('.') if len(s.strip()) > 15]
        
        scored_sentences = []
        for s in sentences:
            s_words = set(re.findall(r'\b\w{3,}\b', s.lower()))
            overlap = len(query_words.intersection(s_words))
            scored_sentences.append((overlap, s))
        
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        relevant_text = scored_sentences[0][1] if scored_sentences else sentences[0] if sentences else context_text[:200]
        
        # Extract figures
        numbers = re.findall(r'\$[\d,.]+\s*(?:billion|million|trillion)?|\d+\.?\d*%', context_text)
        figures_clause = f" Key figures include: {', '.join(numbers[:3])}." if numbers else ""
        
        response = f"According to the SEC filing [Source {doc_id} | {ticker}], {relevant_text}.{figures_clause}"
        return response

    def run(self, user_query: str) -> dict:
        """
        Execute the full 3-stage inference pipeline.
        
        Args:
            user_query: Raw user query string.
            
        Returns:
            Dict with status, answer, grounding_score, and sources.
        """
        # STAGE 1: Pre-Inference Safety Guardrail
        pre_check = self.guardrails.pre_inference_check(user_query)
        if not pre_check.passed:
            return {
                "status": "blocked_pre_inference",
                "answer": pre_check.reason,
                "grounding_score": 0.0,
                "sources": [],
            }

        # STAGE 2: ChromaDB RAG Retrieval + LLaMA 3 8B Inference
        chunks = self.rag.retrieve_context(pre_check.sanitized_text)
        raw_answer = self.generate_llm_answer(pre_check.sanitized_text, chunks)

        # STAGE 3: Post-Inference Grounding & Compliance Guardrail
        post_check = self.guardrails.post_inference_check(
            raw_answer, chunks, threshold=self.cfg.grounding_threshold
        )
        return {
            "status": "success" if post_check.passed else "blocked_post_inference",
            "answer": post_check.sanitized_text if post_check.sanitized_text else raw_answer,
            "grounding_score": post_check.grounding_score,
            "sources": [c["doc_id"] for c in chunks],
        }
