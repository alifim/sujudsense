import os
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    # Embedding & DB
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    persist_directory: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "3"))
    
    # LLM Settings
    fast_llm_model: str = os.getenv("FAST_LLM_MODEL", "openai/gpt-oss-120b")
    fast_llm_model_max_tokens: int = int(os.getenv("FAST_LLM_MAX_TOKENS", "256"))
    heavy_llm_model: str = os.getenv("HEAVY_LLM_MODEL", "openai/gpt-oss-120b")
    heavy_llm_temperature: float = float(os.getenv("HEAVY_LLM_TEMPERATURE", "0.1"))
    heavy_llm_max_tokens: int = int(os.getenv("HEAVY_LLM_MAX_TOKENS", "512"))
    
    # Security Firewall
    firewall_threshold: float = float(os.getenv("FIREWALL_THRESHOLD", "1.4"))

config = AppConfig()