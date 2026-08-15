import os
from dotenv import load_dotenv
import torch

load_dotenv()

# ---------- Paths ----------
STORAGE_DIR = os.environ.get("STORAGE_DIR", "./storage")

# ---------- Models ----------
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
HYDE_MODEL_NAME = os.environ.get("HYDE_MODEL_NAME", "google/flan-t5-small")
CROSS_ENCODER_NAME = os.environ.get("CROSS_ENCODER_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------- LLM Provider ----------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# ---------- Retrieval ----------
TOP_K = int(os.environ.get("TOP_K", 35))            # number of candidates to retrieve before reranking
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", 8))   # number of chunks to include in the prompt
CHUNK_TRUNCATE = int(os.environ.get("CHUNK_TRUNCATE", 1500))  # max chars per chunk in prompt

# HyDE (Hypothetical Document Embeddings)
# For near-zero hallucination, it is recommended to set this to False,
# as HyDE can introduce external knowledge into the retrieval query.
HYDE_ENABLED = os.environ.get("HYDE_ENABLED", "false").lower() == "true"

# ---------- Evidence Gate ----------
# If the average cross-encoder score of the top contexts is below this threshold,
# the system will abstain instead of calling the LLM.
EVIDENCE_THRESHOLD = float(os.environ.get("EVIDENCE_THRESHOLD", 0.35))
TOPK_THRESHOLD = float(os.environ.get("TOPK_THRESHOLD", 0.3))

# ---------- Claim Verification (NLI) ----------
# Toggle to enable/disable verification of generated claims against the retrieved context.
ENABLE_NLI_VERIFICATION = os.environ.get("ENABLE_NLI_VERIFICATION", "true").lower() == "true"
# NLI model to use for verification (should be a cross-encoder fine-tuned for NLI)
NLI_MODEL_NAME = os.environ.get("NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-base")
# Minimum average entailment score required to accept the answer.
NLI_THRESHOLD = float(os.environ.get("NLI_THRESHOLD", 0.5))

# ---------- Cloudflare Tunnel ----------
ENABLE_TUNNEL = os.environ.get("ENABLE_TUNNEL", "false").lower() == "true"

# ---------- Device ----------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'