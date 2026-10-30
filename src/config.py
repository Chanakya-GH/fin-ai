"""
Central configuration for the AI Financial Advisory Agent.
Manages hyperparameters, model paths, and runtime settings.
"""
from dataclasses import dataclass, field
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class AgentConfig:
    """Configuration for the Financial Advisory Agent pipeline."""
    
    # Model settings
    base_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_dir: str = str(PROJECT_ROOT / "checkpoints" / "llama3-8b-finance-qlora")
    
    # RAG settings
    chroma_persist_dir: str = str(PROJECT_ROOT / "chroma_db")
    chroma_collection: str = "sec_10k_filings"
    embedding_model_id: str = str(PROJECT_ROOT / "models" / "bge-base-en-v1.5") if (PROJECT_ROOT / "models" / "bge-base-en-v1.5").exists() else "BAAI/bge-base-en-v1.5"
    top_k_retrieval: int = 4
    
    # Safety settings
    grounding_threshold: float = 0.65
    
    # Generation settings
    max_new_tokens: int = 384
    temperature: float = 0.1
    
    # Runtime mode
    demo_mode: bool = field(default_factory=lambda: os.environ.get("DEMO_MODE", "false").lower() == "true")
    
    # Data paths
    data_dir: str = str(PROJECT_ROOT / "data")
    raw_data_dir: str = str(PROJECT_ROOT / "data" / "raw")
    processed_data_dir: str = str(PROJECT_ROOT / "data" / "processed")

    def __post_init__(self):
        """Create necessary directories."""
        for d in [self.data_dir, self.raw_data_dir, self.processed_data_dir,
                  self.chroma_persist_dir, Path(self.adapter_dir).parent]:
            Path(d).mkdir(parents=True, exist_ok=True)
