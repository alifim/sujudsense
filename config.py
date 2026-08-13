import os
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class AppConfig:
    # Embedding & DB
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    persist_directory: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "10"))
    final_k: int = int(os.getenv("FINAL_K", "3"))
    use_reranker: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_hybrid: bool = os.getenv("USE_HYBRID", "false").lower() == "true"
    hybrid_weights: List[float] = field(default_factory=lambda: [
        float(os.getenv("HYBRID_BM25_WEIGHT", "0.5")),
        float(os.getenv("HYBRID_DENSE_WEIGHT", "0.5")),
    ])
    
    # LLM Settings
    fast_llm_model: str = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")
    fast_llm_model_max_tokens: int = int(os.getenv("FAST_LLM_MAX_TOKENS", "256"))
    heavy_llm_model: str = os.getenv("HEAVY_LLM_MODEL", "llama-3.3-70b-versatile")
    heavy_llm_temperature: float = float(os.getenv("HEAVY_LLM_TEMPERATURE", "0.1"))
    heavy_llm_max_tokens: int = int(os.getenv("HEAVY_LLM_MAX_TOKENS", "512"))
    
    # Security Firewall
    firewall_threshold: float = float(os.getenv("FIREWALL_THRESHOLD", "1"))

config = AppConfig()