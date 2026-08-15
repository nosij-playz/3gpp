"""
FastAPI server for 3GPP RAG Chatbot
All models are configurable via .env
"""

import os
import pickle
import json
import numpy as np
import torch
from contextlib import asynccontextmanager
from typing import List

import faiss
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG (all from .env) ----------
STORAGE_DIR = os.environ.get("STORAGE_DIR", "./storage")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
HYDE_MODEL_NAME = os.environ.get("HYDE_MODEL_NAME", "google/flan-t5-small")
CROSS_ENCODER_NAME = os.environ.get("CROSS_ENCODER_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
TOP_K = int(os.environ.get("TOP_K", 25))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")
print(f"Embedding model: {EMBED_MODEL_NAME}")
print(f"HyDE model: {HYDE_MODEL_NAME}")
print(f"Cross-encoder: {CROSS_ENCODER_NAME}")
print(f"Groq model: {GROQ_MODEL}")
print(f"Top K: {TOP_K}")

# ---------- GLOBAL VARIABLES ----------
index = None
nodes = None
texts = None
embed_model = None
hyde_tokenizer = None
hyde_model = None
cross_encoder = None
groq_client = None

# ---------- QUERY EXPANSION (for 3GPP abbreviations) ----------
def expand_query(query: str) -> str:
    """Add common expansions for 3GPP abbreviations."""
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
    global index, nodes, texts, embed_model, hyde_tokenizer, hyde_model, cross_encoder, groq_client

    print("🔧 Loading RAG memory...")
    index = faiss.read_index(os.path.join(STORAGE_DIR, "faiss.index"))
    with open(os.path.join(STORAGE_DIR, "nodes.pkl"), "rb") as f:
        nodes = pickle.load(f)
    with open(os.path.join(STORAGE_DIR, "texts.pkl"), "rb") as f:
        texts = pickle.load(f)
    print(f"✅ Memory loaded: {len(texts)} chunks, index dim {index.d}")

    print("🔧 Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

    print("🔧 Loading HyDE model...")
    hyde_tokenizer = AutoTokenizer.from_pretrained(HYDE_MODEL_NAME)
    hyde_model = AutoModelForSeq2SeqLM.from_pretrained(HYDE_MODEL_NAME).to('cpu')
    hyde_model.eval()

    print("🔧 Loading cross-encoder...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device=DEVICE)

    print("🔧 Initializing Groq client...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment")
    groq_client = Groq(api_key=api_key)

    print("🚀 Server ready.")
    yield

app = FastAPI(title="3GPP RAG Bot", lifespan=lifespan)

# ---------- PYDANTIC ----------
class QueryRequest(BaseModel):
    question: str
    stream: bool = False

class Source(BaseModel):
    text: str
    metadata: dict
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    hallucination_risk: bool = False

# ---------- RETRIEVAL FUNCTIONS ----------
def generate_hypothesis(query: str) -> str:
    prompt = f"Write a detailed answer to this question: {query}"
    inputs = hyde_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hyde_model.generate(**inputs, max_new_tokens=150)
    return hyde_tokenizer.decode(outputs[0], skip_special_tokens=True)

def hybrid_retrieve(query: str, top_k: int = TOP_K) -> List[dict]:
    # Expand query
    expanded_query = expand_query(query)
    
    # HyDE
    hyde_doc = generate_hypothesis(expanded_query)
    combined_query = expanded_query + " " + hyde_doc

    # FAISS
    q_emb = embed_model.encode([combined_query], convert_to_tensor=True)
    q_emb = q_emb.cpu().numpy().astype('float32')
    distances, indices = index.search(q_emb, 50)

    # Cross-encoder re-rank
    candidate_texts = [texts[i] for i in indices[0]]
    pairs = [[query, doc] for doc in candidate_texts]
    scores = cross_encoder.predict(pairs)

    # Sort and select top_k
    sorted_idx = np.argsort(scores)[::-1][:top_k]
    top_indices = [indices[0][i] for i in sorted_idx]
    top_scores = [scores[i] for i in sorted_idx]

    # Normalise scores with softmax (0–1) – for display only
    exp_scores = np.exp(top_scores - np.max(top_scores))
    softmax_scores = exp_scores / np.sum(exp_scores)

    contexts = []
    for i, idx in enumerate(top_indices):
        contexts.append({
            "text": texts[idx],
            "metadata": nodes[idx].metadata,
            "score": float(softmax_scores[i])
        })
    return contexts

def build_prompt(query: str, contexts: List[dict]) -> str:
    context_text = "\n\n".join([
        f"[Source: {ctx['metadata'].get('file_name', 'unknown')}]\n{ctx['text']}"
        for ctx in contexts
    ])
    system = (
        "You are a 3GPP standards expert. Answer the user's question using ONLY the provided context.\n"
        "If the answer is spread across multiple chunks, COMBINE the information from ALL relevant chunks.\n"
        "If the context is completely irrelevant to the question, respond with: "
        "'I cannot find this information in the provided documents.'\n"
        "For every factual claim, provide a citation in the format [Source: filename].\n"
        "Present structured information as a table or bullet list.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    )
    return system

# ---------- API ENDPOINTS ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        contexts = hybrid_retrieve(req.question)
        prompt = build_prompt(req.question, contexts)

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
            top_p=1,
            stream=False,
        )
        answer = completion.choices[0].message.content

        sources = [
            {"text": c["text"], "metadata": c["metadata"], "score": c["score"]}
            for c in contexts[:5]
        ]
        return {"answer": answer, "sources": sources, "hallucination_risk": False}

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
            contexts = hybrid_retrieve(req.question)
            prompt = build_prompt(req.question, contexts)

            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
                top_p=1,
                stream=True,
            )
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)