"""
RAG Engine Module using ChromaDB + BGE Embeddings + LlamaIndex.

Manages the vector store for SEC 10-K filing passages, handles document
ingestion, and provides semantic retrieval for the inference pipeline.
"""
import json
from pathlib import Path

import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import AgentConfig


class FinancialRAGEngine:
    """
    RAG engine for financial document retrieval.
    
    Uses ChromaDB as the persistent vector store with BGE embeddings
    for semantic similarity search over SEC 10-K filing passages.
    """
    
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        
        # Configure LlamaIndex settings
        Settings.embed_model = HuggingFaceEmbedding(model_name=cfg.embedding_model_id)
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(cfg.chroma_collection)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store, storage_context=self.storage_context
        )

    def ingest_10k_corpus(self, split: str = "train[5000:6000]"):
        """
        Index SEC 10-K passages into ChromaDB if collection is empty.
        
        Can ingest from either HuggingFace dataset split or local JSONL file.
        
        Args:
            split: HuggingFace dataset split string (used when loading from HF).
        """
        if self.collection.count() > 0:
            print(f"ChromaDB collection '{self.cfg.chroma_collection}' already has "
                  f"{self.collection.count()} documents. Skipping ingestion.")
            return
        
        docs = []
        
        # Try loading from local processed file first
        local_corpus = Path(self.cfg.processed_data_dir) / "rag_corpus.jsonl"
        if local_corpus.exists():
            print(f"Loading RAG corpus from local file: {local_corpus}")
            with open(local_corpus) as f:
                for line in f:
                    row = json.loads(line.strip())
                    docs.append(
                        Document(
                            text=row["text"],
                            metadata={
                                "ticker": row.get("ticker", "UNKNOWN"),
                                "filing": row.get("filing", "2023_10K"),
                                "doc_id": row.get("doc_id", f"sec_{len(docs)}"),
                            },
                        )
                    )
        else:
            # Fall back to HuggingFace download
            print(f"Loading RAG corpus from HuggingFace: virattt/financial-qa-10K [{split}]")
            from datasets import load_dataset
            ds = load_dataset("virattt/financial-qa-10K", split=split)
            for idx, row in enumerate(ds):
                docs.append(
                    Document(
                        text=row["context"],
                        metadata={
                            "ticker": row.get("ticker", "UNKNOWN"),
                            "filing": row.get("filing", "2023_10K"),
                            "doc_id": f"sec_{idx}",
                        },
                    )
                )
        
        if not docs:
            print("WARNING: No documents found for RAG corpus ingestion!")
            return
            
        print(f"Ingesting {len(docs)} documents into ChromaDB...")
        self.index = VectorStoreIndex.from_documents(
            docs, storage_context=self.storage_context, show_progress=True
        )
        print(f"Ingestion complete. Collection now has {self.collection.count()} documents.")

    def retrieve_context(self, query: str) -> list[dict]:
        """
        Retrieve relevant financial context chunks for a query.
        
        Args:
            query: User's financial question.
            
        Returns:
            List of dicts with doc_id, ticker, score, and text fields.
        """
        retriever = self.index.as_retriever(similarity_top_k=self.cfg.top_k_retrieval)
        nodes = retriever.retrieve(query)
        return [
            {
                "doc_id": n.metadata.get("doc_id", "N/A"),
                "ticker": n.metadata.get("ticker", "N/A"),
                "score": float(n.score or 0.0),
                "text": n.node.get_content(),
            }
            for n in nodes
        ]
