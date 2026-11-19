"""Central settings, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

        # Chat and embeddings can use different providers (e.g. Groq for chat, Ollama for embeddings).
    # Leave a value unset to fall back to OPENAI_API_KEY / OPENAI_BASE_URL.
    chat_api_key: str | None = os.getenv("CHAT_API_KEY") or None
    chat_base_url: str | None = os.getenv("CHAT_BASE_URL") or None
    embed_api_key: str | None = os.getenv("EMBED_API_KEY") or None
    embed_base_url: str | None = os.getenv("EMBED_BASE_URL") or None

    # Models: check the OpenAI docs for current names and override via env if needed.
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1536"))
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    # Chunking (in words). Re-ingest after changing, then re-run the eval to compare.
    chunk_words: int = int(os.getenv("CHUNK_WORDS", "200"))
    overlap_words: int = int(os.getenv("OVERLAP_WORDS", "30"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))
    candidate_k: int = int(os.getenv("CANDIDATE_K", "20"))
    # Cosine similarity of the best chunk below this => refuse without calling the LLM.
    # This is a starting point: tune it with the eval set.
    min_vector_score: float = float(os.getenv("MIN_VECTOR_SCORE", "0.25"))
        # Weight of keyword results when fusing with vector results (1.0 = equal, lower = trust vectors more).
    keyword_weight: float = float(os.getenv("KEYWORD_WEIGHT", "1.0"))
    use_reranker: bool = _bool("USE_RERANKER", False)
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Public-demo cost protection
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "8"))
    daily_request_cap: int = int(os.getenv("DAILY_REQUEST_CAP", "300"))


settings = Settings()
