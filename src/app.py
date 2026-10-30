"""
FastAPI Production Server for the AI Financial Advisory Agent.

Exposes the 3-Stage RAG + QLoRA LLaMA 3 8B pipeline via REST API.

Usage:
    # Full mode (requires GPU + trained adapter):
    uvicorn src.app:app --host 0.0.0.0 --port 8000
    
    # Demo mode (no GPU required):
    DEMO_MODE=true uvicorn src.app:app --host 0.0.0.0 --port 8000
"""
import json
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import AgentConfig
from pipeline import FinancialAdvisoryPipeline
from finetune_qlora import calculate_trainable_parameters


# --- Pydantic Schemas ---

class QueryRequest(BaseModel):
    """Request schema for the /v1/advise endpoint."""
    query: str = Field(..., min_length=3, max_length=1000, 
                       description="Financial question to the advisory agent")

class QueryResponse(BaseModel):
    """Response schema for the /v1/advise endpoint."""
    status: str = Field(..., description="Pipeline result status")
    answer: str = Field(..., description="Generated advisory response")
    grounding_score: float = Field(..., description="Source grounding verification score (0-1)")
    sources: list[str] = Field(default_factory=list, description="Referenced source document IDs")
    latency_ms: float = Field(0.0, description="End-to-end pipeline latency in milliseconds")

class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    mode: str
    rag_documents: int
    model_loaded: bool


# --- App Lifecycle ---

pipeline: FinancialAdvisoryPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup, cleanup on shutdown."""
    global pipeline
    cfg = AgentConfig()
    
    # In demo mode, don't try to load the LLM
    load_llm = not cfg.demo_mode
    pipeline = FinancialAdvisoryPipeline(cfg, load_llm=load_llm)
    pipeline.rag.ingest_10k_corpus()
    
    print(f"\n{'='*60}")
    print(f"Financial Advisory Agent API started")
    print(f"Mode: {'DEMO' if cfg.demo_mode else 'PRODUCTION'}")
    print(f"RAG documents: {pipeline.rag.collection.count()}")
    print(f"{'='*60}\n")
    
    yield
    
    # Cleanup
    pipeline = None

app = FastAPI(
    title="AI-Powered Financial Advisory Agent",
    version="1.0.0",
    description=(
        "3-Stage RAG + QLoRA LLaMA 3 8B Financial Agent with Pre/Post Safety Guardrails. "
        "Provides grounded, compliant financial advisory responses from SEC 10-K filings."
    ),
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
def index_page():
    """Interactive visual dashboard for demonstration."""
    html_path = Path(__file__).resolve().parent / "templates" / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>AI-Powered Financial Advisory Agent API</h1><p>Visit /docs for OpenAPI documentation.</p>")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return HealthResponse(
        status="healthy",
        mode="demo" if pipeline._demo_mode else "production",
        rag_documents=pipeline.rag.collection.count(),
        model_loaded=pipeline.model is not None,
    )


@app.get("/v1/benchmark")
def get_benchmark_results():
    """Return the latest 100-case evaluation benchmark results."""
    results_file = Path(__file__).resolve().parent.parent / "eval" / "results" / "eval_results_latest.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            return json.load(f)
    return {"message": "No benchmark results found. Run python -m eval.evaluate_100_cases"}


@app.get("/v1/parameter-math")
def get_parameter_math():
    """Return the exact mathematical parameter derivation for 0.52% QLoRA."""
    return calculate_trainable_parameters()


@app.post("/v1/advise", response_model=QueryResponse)
def advise(req: QueryRequest):
    """
    Main advisory endpoint.
    
    Processes a financial question through the 3-stage pipeline:
    1. Pre-inference guardrail (adversarial detection, PII masking)
    2. RAG retrieval + LLM generation
    3. Post-inference grounding verification
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start = time.perf_counter()
    result = pipeline.run(req.query)
    latency_ms = (time.perf_counter() - start) * 1000
    
    return QueryResponse(
        status=result["status"],
        answer=result["answer"],
        grounding_score=result["grounding_score"],
        sources=result["sources"],
        latency_ms=round(latency_ms, 2),
    )


@app.get("/v1/sources")
def list_sources():
    """List available source documents in the RAG knowledge base."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "collection": pipeline.cfg.chroma_collection,
        "total_documents": pipeline.rag.collection.count(),
        "embedding_model": pipeline.cfg.embedding_model_id,
    }
