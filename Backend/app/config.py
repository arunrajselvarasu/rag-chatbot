"""Configuration: environment variables (.env) and tunable constants."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# The one fallback used everywhere an answer cannot be supported by the documents.
FALLBACK_MESSAGE = (
    "I'm sorry, I can only answer questions related to the uploaded documents. "
    "The available document context does not provide enough relevant information "
    "to answer this question."
)

# Internal FAISS persistence (deliberately not an environment variable).
INDEX_DIR = BASE_DIR / "data" / "faiss"

# Tunable constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
RELEVANCE_THRESHOLD = 0.30   # minimum cosine similarity for a chunk to count as relevant
MAX_QUESTION_CHARS = 1000
MAX_ANSWER_CHARS = 6000
HISTORY_LIMIT = 6            # messages passed to the LLM
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    embedding_model: str


def load_settings() -> Settings:
    """Read required variables; fail fast at startup if any is missing."""
    names = ("GROQ_API_KEY", "GROQ_MODEL", "EMBEDDING_MODEL")
    values = {n: os.getenv(n, "").strip() for n in names}
    missing = [n for n, v in values.items() if not v or v.startswith("your_")]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s) in Backend/.env: " + ", ".join(missing)
        )
    return Settings(values["GROQ_API_KEY"], values["GROQ_MODEL"], values["EMBEDDING_MODEL"])
