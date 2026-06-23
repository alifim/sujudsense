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
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    
    # Security Firewall
    firewall_threshold: float = float(os.getenv("FIREWALL_THRESHOLD", "1.4"))

config = AppConfig()