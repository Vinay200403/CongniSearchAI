"""
Central configuration for CogniSearch AI.
All tunables live here so the rest of the codebase never hardcodes paths
or magic numbers.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Storage paths -----------------------------------------------------
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- Upload constraints --------------------------------------------------
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "md"}
MAX_CONTENT_LENGTH_MB = 20

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE_CHARS = 1000       # characters per chunk
CHUNK_OVERLAP_CHARS = 150     # overlap between consecutive chunks

# --- Retrieval ---------------------------------------------------------
TOP_K = 4                      # number of chunks retrieved per query

# --- Embeddings & generation ---------------------------------------------
# CogniSearch supports three interchangeable backends, auto-selected in
# this priority order:
#   1. Gemini   — if GEMINI_API_KEY is set (get one free at aistudio.google.com)
#   2. OpenAI   — if OPENAI_API_KEY is set
#   3. Local    — zero-cost TF-IDF fallback, always available
#
# Note: a Gemini *consumer* subscription (the Gemini app / "Gemini Pro")
# is a different product from Gemini API access. GEMINI_API_KEY must come
# from Google AI Studio (aistudio.google.com), which has its own free tier.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
GEMINI_EMBED_DIM = 768  # requested via output_dimensionality; model default is 3072

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"

USE_GEMINI = bool(GEMINI_API_KEY)
USE_OPENAI = bool(OPENAI_API_KEY) and not USE_GEMINI

def active_backend():
    if USE_GEMINI:
        return "gemini"
    if USE_OPENAI:
        return "openai"
    return "local"

# --- Flask ---------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
