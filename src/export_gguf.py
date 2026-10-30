"""
GGUF Export & CPU Inference Module for Financial Advisory Agent.

Supports exporting fine-tuned QLoRA LLaMA 3 8B models to GGUF format
for lightweight CPU inference using llama.cpp or Ollama (Roadmap Section 5.1).

Quantization Target: Q4_K_M (4-bit Medium Quantization, ~4.8 GB RAM)

Usage:
    # Generate Ollama Modelfile and export instructions:
    python src/export_gguf.py --generate-modelfile
    
    # Merge LoRA weights and export (requires GPU / base model):
    python src/export_gguf.py --merge
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import AgentConfig


SYSTEM_PROMPT = (
    "You are a compliant AI Financial Advisory Agent. Answer user questions accurately "
    "using ONLY the provided financial context. Cite specific figures from the context. "
    "If the context does not contain the answer, state that insufficient data is available "
    "rather than speculating."
)

MODELFILE_TEMPLATE = """# Ollama Modelfile for AI Financial Advisory Agent
# Quantized GGUF Q4_K_M model for local CPU inference

FROM ./checkpoints/llama3-8b-finance.Q4_K_M.gguf

TEMPLATE \"\"\"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

\"\"\"

SYSTEM \"\"\"You are a compliant AI Financial Advisory Agent. Answer user questions accurately using ONLY the provided financial context. Cite specific figures from the context. If the context does not contain the answer, state that insufficient data is available rather than speculating.\"\"\"

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.1
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
"""


def generate_modelfile(cfg: AgentConfig):
    """Generate Ollama Modelfile and print instructions for llama.cpp compilation."""
    modelfile_path = PROJECT_ROOT / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(MODELFILE_TEMPLATE)
    
    print("=" * 70)
    print("GGUF & CPU INFERENCE EXPORT GUIDE (Roadmap Section 5.1)")
    print("=" * 70)
    print(f"✓ Ollama Modelfile generated at: {modelfile_path}\n")
    print("Step-by-step workflow for CPU Quantization & Deployment:")
    print("-" * 70)
    print("1. Merge LoRA weights into base model (FP16):")
    print("   python src/export_gguf.py --merge\n")
    print("2. Convert FP16 HuggingFace model to GGUF format:")
    print("   python llama.cpp/convert_hf_to_gguf.py ./checkpoints/llama3-8b-finance-merged \\")
    print("       --outfile ./checkpoints/llama3-8b-finance-f16.gguf --outtype f16\n")
    print("3. Quantize to Q4_K_M (4-bit medium for CPU execution, ~4.8 GB RAM):")
    print("   ./llama.cpp/llama-quantize ./checkpoints/llama3-8b-finance-f16.gguf \\")
    print("       ./checkpoints/llama3-8b-finance.Q4_K_M.gguf Q4_K_M\n")
    print("4. Create and run the Ollama local model on CPU:")
    print("   ollama create fin-advisory-agent -f Modelfile")
    print("   ollama run fin-advisory-agent \"What was NVIDIA's gross margin in 2024?\"")
    print("=" * 70)
    return str(modelfile_path)


def merge_lora_weights(cfg: AgentConfig):
    """Merge LoRA weights with base model and save merged model in FP16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    merged_dir = Path(cfg.adapter_dir).parent / "llama3-8b-finance-merged"
    print(f"Loading base model: {cfg.base_model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print(f"Loading PEFT adapter: {cfg.adapter_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, cfg.adapter_dir)
    
    print("Merging adapter weights with base model...")
    merged_model = peft_model.merge_and_unload()
    
    print(f"Saving merged FP16 model to: {merged_dir}...")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print("Merge complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuned model for CPU GGUF inference")
    parser.add_argument("--generate-modelfile", action="store_true", help="Generate Ollama Modelfile")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA weights with base model")
    
    args = parser.parse_args()
    cfg = AgentConfig()
    
    if args.merge:
        merge_lora_weights(cfg)
    else:
        generate_modelfile(cfg)
