from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

class Settings(BaseSettings):
    app_name: str = "Enterprise IT Support Agentic RAG Copilot"
    app_env: str = "development"
    groq_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "industry-agentic-rag-kb"
    pinecone_namespace: str = "langgraph-agentic-rag"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "openai/gpt-oss-20b"
    judge_model: str = "openai/gpt-oss-120b"
    top_k: int = 4
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    bm25_refresh_seconds: int = 300
    max_retries: int = 1
    admin_api_key: str = "change-me"
    database_url: str = "postgresql://support:support@localhost:5432/support_agent"
    history_messages: int = 6
    upload_dir: str = str(BASE_DIR / "uploads")
    models_dir: str = str(BASE_DIR / ".cache" / "models")
    sample_kb_dir: str = str(BASE_DIR / "data")

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")




@lru_cache
def get_settings() -> Settings:
    return Settings()
