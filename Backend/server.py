"""
Evidence-Controlled RAG Server – Supports Groq & Ollama
- LLM provider configurable via .env
- Evidence gate before generation
- Streaming and non‑streaming endpoints
"""

import os
import pickle
import json
import numpy as np
import torch
import re
import requests
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple, Generator, Dict, Any

import faiss
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
STORAGE_DIR = os.environ.get("STORAGE_DIR", "./storage")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
HYDE_MODEL_NAME = os.environ.get("HYDE_MODEL_NAME", "google/flan-t5-small")
CROSS_ENCODER_NAME = os.environ.get("CROSS_ENCODER_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM Provider
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

TOP_K = int(os.environ.get("TOP_K", 35))
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", 8))
EVIDENCE_THRESHOLD = float(os.environ.get("EVIDENCE_THRESHOLD", 0.5))
TOPK_THRESHOLD = float(os.environ.get("TOPK_THRESHOLD", 0.3))
CHUNK_TRUNCATE = int(os.environ.get("CHUNK_TRUNCATE", 1500))
HYDE_ENABLED = os.environ.get("HYDE_ENABLED", "true").lower() == "true"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🧠 LLM Provider: {LLM_PROVIDER}")
if LLM_PROVIDER == "groq":
    print(f"   Model: {GROQ_MODEL}")
else:
    print(f"   Model: {OLLAMA_MODEL} (via {OLLAMA_BASE_URL})")

# ---------- GLOBAL VARIABLES ----------
index = None
nodes = None
texts = None
embed_model = None
hyde_tokenizer = None
hyde_model = None
cross_encoder = None

# LLM client (initialized lazily)
llm_client = None

# ---------- QUERY EXPANSION ----------
def expand_query(query: str) -> str:
    expansions = {
        "AMF": "Access and Mobility Management Function",
        "SMF": "Session Management Function",
        "UPF": "User Plane Function",
        "NR": "New Radio",
        "gNB": "next-generation NodeB",
        "QoS": "Quality of Service",
        "QoE": "Quality of Experience",
        "NSA": "Non-Standalone",
        "SA": "Standalone",
        "5G": "5th Generation",
        "5GC": "5G Core",
        "NGAP": "Next Generation Application Protocol",
        "NG-RAN": "Next Generation Radio Access Network",
        "PDU": "Protocol Data Unit",
        "SDAP": "Service Data Adaptation Protocol",
        "RRC": "Radio Resource Control",
        "NAS": "Non-Access Stratum",
    }
    words = query.split()
    expanded = []
    for w in words:
        found = False
        for key, val in expansions.items():
            if key in w:
                expanded.append(f"{w} ({val})")
                found = True
                break
        if not found:
            expanded.append(w)
    return " ".join(expanded)

# ---------- LIFECYCLE ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, nodes, texts, embed_model, hyde_tokenizer, hyde_model, cross_encoder, llm_client

    print("🔧 Loading memory...")
    index = faiss.read_index(os.path.join(STORAGE_DIR, "faiss.index"))
    with open(os.path.join(STORAGE_DIR, "nodes.pkl"), "rb") as f:
        nodes = pickle.load(f)
    with open(os.path.join(STORAGE_DIR, "texts.pkl"), "rb") as f:
        texts = pickle.load(f)
    print(f"✅ Memory loaded: {len(texts)} chunks")

    print("🔧 Loading retrieval models...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    hyde_tokenizer = AutoTokenizer.from_pretrained(HYDE_MODEL_NAME)
    hyde_model = AutoModelForSeq2SeqLM.from_pretrained(HYDE_MODEL_NAME).to('cpu')
    hyde_model.eval()
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device=DEVICE)

    # Initialize LLM client based on provider
    print("🔧 Initializing LLM client...")
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set")
        from groq import Groq
        llm_client = Groq(api_key=GROQ_API_KEY)
    elif LLM_PROVIDER == "ollama":
        # No client needed – we'll use requests directly
        llm_client = None
        # Test connectivity
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            if resp.status_code != 200:
                print(f"⚠️  Ollama not reachable at {OLLAMA_BASE_URL}")
        except:
            print(f"⚠️  Could not connect to Ollama at {OLLAMA_BASE_URL}")
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    print("🚀 Server ready.")
    yield

app = FastAPI(title="3GPP RAG (Evidence-Controlled)", lifespan=lifespan)

# ---------- PYDANTIC MODELS ----------
class QueryRequest(BaseModel):
    question: str
    stream: bool = False

class Source(BaseModel):
    text: str
    metadata: dict
    score: float
    file_name: str = ""

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    abstained: bool = Field(default=False, description="Whether the system refused to answer")
    evidence_score: float = Field(default=0.0, description="Score based on evidence quality (0-1)")
    retrieval_count: int = Field(default=0, description="Number of chunks retrieved")

# ---------- RETRIEVAL FUNCTIONS ----------
def generate_hypothesis(query: str) -> str:
    prompt = f"Write a detailed answer to this question: {query}"
    inputs = hyde_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hyde_model.generate(**inputs, max_new_tokens=150)
    return hyde_tokenizer.decode(outputs[0], skip_special_tokens=True)

def retrieve_evidence(query: str) -> Tuple[List[dict], float, float]:
    expanded_query = expand_query(query)
    if HYDE_ENABLED:
        hyde_doc = generate_hypothesis(expanded_query)
        combined = expanded_query + " " + hyde_doc
    else:
        combined = expanded_query

    q_emb = embed_model.encode([combined], convert_to_tensor=True)
    q_emb = q_emb.cpu().numpy().astype('float32')
    distances, indices = index.search(q_emb, 50)

    candidates = [texts[i] for i in indices[0]]
    pairs = [[query, doc] for doc in candidates]
    scores = cross_encoder.predict(pairs)

    sorted_idx = np.argsort(scores)[::-1]
    top_indices = [indices[0][i] for i in sorted_idx]
    top_scores = [scores[i] for i in sorted_idx]

    seen_texts = set()
    contexts = []
    for i, idx in enumerate(top_indices):
        text = texts[idx]
        if text in seen_texts:
            continue
        seen_texts.add(text)
        file_name = nodes[idx].metadata.get('file_name', 'unknown')
        contexts.append({
            "text": text,
            "metadata": nodes[idx].metadata,
            "score": float(top_scores[i]),
            "file_name": file_name
        })

    if contexts:
        top_score = contexts[0]["score"]
        top_3 = contexts[:3]
        avg_top = sum(c["score"] for c in top_3) / len(top_3) if top_3 else 0.0
    else:
        top_score = 0.0
        avg_top = 0.0

    return contexts, top_score, avg_top

def evidence_is_sufficient(contexts: List[dict], top_score: float, avg_top: float) -> Tuple[bool, float]:
    if not contexts:
        return False, 0.0
    if top_score <= 0:
        return False, 0.0
    if top_score < EVIDENCE_THRESHOLD:
        return False, 0.0
    if avg_top < TOPK_THRESHOLD:
        return False, 0.0
    good_chunks = sum(1 for c in contexts if c["score"] > EVIDENCE_THRESHOLD / 2)
    if good_chunks < 2:
        return False, 0.0

    normalized_top = min(1.0, top_score / 8.0)
    normalized_avg = min(1.0, avg_top / 6.0)
    coverage = min(1.0, good_chunks / 4.0)
    evidence_score = normalized_top * 0.5 + normalized_avg * 0.3 + coverage * 0.2
    return True, evidence_score

# ---------- PROMPT BUILDING ----------
def build_prompt(query: str, contexts: List[dict]) -> str:
    chunks = contexts[:MAX_CHUNKS]
    context_parts = []
    for i, c in enumerate(chunks):
        text = c["text"]
        if len(text) > CHUNK_TRUNCATE:
            cut = text[:CHUNK_TRUNCATE].rfind('. ')
            if cut > CHUNK_TRUNCATE // 2:
                text = text[:cut + 1] + "..."
            else:
                text = text[:CHUNK_TRUNCATE] + "..."

        metadata = c["metadata"]
        file_name = c.get("file_name", metadata.get("file_name", "unknown"))
        spec = metadata.get("specification", "")
        section = metadata.get("section", "")
        release = metadata.get("release", "")

        if spec and section:
            citation = f"{spec}, Release {release}, Clause {section}" if release else f"{spec}, Clause {section}"
        else:
            citation = file_name

        context_parts.append(f"[Source: {citation}]\n{text}")

    context_text = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a 3GPP standards expert with access to a knowledge base.\n\n"
        "You must answer the user's question using ONLY the provided context.\n\n"
        "RULES:\n"
        "1. If the context contains the answer, provide a clear, cited answer.\n"
        "2. If the context does NOT contain the answer, say: 'I cannot find this information in the provided documents.'\n"
        "3. If the question is completely unrelated to 3GPP/telecom, say: 'I can only answer questions about 3GPP standards.'\n"
        "4. For every factual claim, provide a citation in the format [Source: ...].\n"
        "5. Use tables or bullet lists when appropriate.\n"
        "6. Do not add information from outside the context.\n\n"
        "--- PROVIDED CONTEXT ---\n"
        f"{context_text}\n"
        "--- END OF CONTEXT ---\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return system

# ---------- LLM GENERATION (Unified) ----------
def generate_llm_response(prompt: str, stream: bool = False):
    """Generate response from either Groq or Ollama."""
    if LLM_PROVIDER == "groq":
        return _generate_groq(prompt, stream)
    elif LLM_PROVIDER == "ollama":
        return _generate_ollama(prompt, stream)
    else:
        raise ValueError(f"Unknown provider: {LLM_PROVIDER}")

def _generate_groq(prompt: str, stream: bool):
    # Groq expects messages list
    messages = [{"role": "user", "content": prompt}]
    completion = llm_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=1200,
        top_p=1,
        stream=stream,
    )
    if stream:
        # Return generator that yields tokens
        def generator():
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        return generator()
    else:
        return completion.choices[0].message.content

def _generate_ollama(prompt: str, stream: bool):
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
        "temperature": 0.0,
        "max_tokens": 1200,
        "top_p": 1,
    }
    if stream:
        def generator():
            with requests.post(url, json=payload, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
        return generator()
    else:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "")

# ---------- API ENDPOINTS ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        contexts, top_score, avg_top = retrieve_evidence(req.question)
        sufficient, evidence_score = evidence_is_sufficient(contexts, top_score, avg_top)

        if not sufficient:
            return {
                "answer": "I cannot find sufficient information to answer this question in the provided 3GPP documents.",
                "sources": [],
                "abstained": True,
                "evidence_score": evidence_score,
                "retrieval_count": len(contexts)
            }

        prompt = build_prompt(req.question, contexts)
        answer = generate_llm_response(prompt, stream=False)

        # Double-check LLM refusal
        refusal_phrases = [
            "cannot find this information",
            "not in the provided documents",
            "only answer questions about 3gpp",
            "outside the scope"
        ]
        llm_refused = any(p in answer.lower() for p in refusal_phrases)
        if llm_refused:
            return {
                "answer": answer,
                "sources": [],
                "abstained": True,
                "evidence_score": evidence_score,
                "retrieval_count": len(contexts)
            }

        sources = [
            Source(
                text=c["text"][:500],
                metadata=c["metadata"],
                score=c["score"],
                file_name=c.get("file_name", c["metadata"].get("file_name", "unknown"))
            )
            for c in contexts[:5]
        ]

        return {
            "answer": answer,
            "sources": sources,
            "abstained": False,
            "evidence_score": evidence_score,
            "retrieval_count": len(contexts)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    if not req.stream:
        return await query_endpoint(req)

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def event_generator():
        try:
            contexts, top_score, avg_top = retrieve_evidence(req.question)
            sufficient, evidence_score = evidence_is_sufficient(contexts, top_score, avg_top)

            if not sufficient:
                yield f"data: {json.dumps({'token': 'I cannot find sufficient information to answer this question in the provided 3GPP documents.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            prompt = build_prompt(req.question, contexts)
            gen = generate_llm_response(prompt, stream=True)
            for token in gen:
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)