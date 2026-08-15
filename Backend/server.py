"""
Pure Brain + Memory RAG Server – Near-Zero Hallucination Edition
- LLM provider configurable via .env (Groq / Ollama)
- Evidence gate: abstains if retrieved context is weak
- Claim verification: NLI model checks generated answer against evidence
- Streaming and non-streaming endpoints
- Optional Cloudflare tunnel for quick public access
"""

import os
import pickle
import json
import re
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple, Generator, Dict, Any

import faiss
import numpy as np
import torch
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# Import all config settings (make sure config.py is in the same directory)
from config import (
    STORAGE_DIR,
    EMBED_MODEL_NAME,
    HYDE_MODEL_NAME,
    CROSS_ENCODER_NAME,
    NLI_MODEL_NAME,
    ENABLE_NLI_VERIFICATION,
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K,
    MAX_CHUNKS,
    CHUNK_TRUNCATE,
    HYDE_ENABLED,
    EVIDENCE_THRESHOLD,
    TOPK_THRESHOLD,
    NLI_THRESHOLD,
    ENABLE_TUNNEL,
    DEVICE
)

load_dotenv()

print(f"🧠 LLM Provider: {LLM_PROVIDER}")
if LLM_PROVIDER == "groq":
    print(f"   Model: {GROQ_MODEL}")
else:
    print(f"   Model: {OLLAMA_MODEL} (via {OLLAMA_BASE_URL})")
if ENABLE_NLI_VERIFICATION:
    print(f"🔍 NLI Verification: enabled (model: {NLI_MODEL_NAME})")
else:
    print("🔍 NLI Verification: disabled")

# ---------- GLOBAL VARIABLES ----------
index = None
nodes = None
texts = None
embed_model = None
hyde_tokenizer = None
hyde_model = None
cross_encoder = None
nli_model = None          # NEW: for claim verification
llm_client = None
TUNNEL_URL = None

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

# ---------- CLOUDFLARE TUNNEL ----------
def start_cloudflare_tunnel():
    global TUNNEL_URL
    if not ENABLE_TUNNEL:
        return
    try:
        cmd = ["cloudflared", "tunnel", "--url", "http://localhost:8000"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        print("🔄 Starting Cloudflare tunnel...")
        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"[cloudflared] {line}")
            match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
            if match:
                TUNNEL_URL = match.group(0)
                print(f"🔗 Tunnel URL: {TUNNEL_URL}")
                break
        process.wait()
    except FileNotFoundError:
        print("⚠️ cloudflared not found. Please install it: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation")
    except Exception as e:
        print(f"⚠️ Error starting Cloudflare tunnel: {e}")

# ---------- LIFECYCLE ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, nodes, texts, embed_model, hyde_tokenizer, hyde_model, cross_encoder, nli_model, llm_client

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

    if ENABLE_NLI_VERIFICATION:
        print("🔧 Loading NLI verification model...")
        nli_model = CrossEncoder(NLI_MODEL_NAME, device=DEVICE)

    print("🔧 Initializing LLM client...")
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set")
        from groq import Groq
        llm_client = Groq(api_key=GROQ_API_KEY)
    elif LLM_PROVIDER == "ollama":
        llm_client = None
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            if resp.status_code != 200:
                print(f"⚠️  Ollama not reachable at {OLLAMA_BASE_URL}")
        except:
            print(f"⚠️  Could not connect to Ollama at {OLLAMA_BASE_URL}")
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    if ENABLE_TUNNEL:
        thread = threading.Thread(target=start_cloudflare_tunnel, daemon=True)
        thread.start()
        time.sleep(1)

    print("🚀 Server ready.")
    yield

app = FastAPI(title="3GPP RAG (Near-Zero Hallucination)", lifespan=lifespan)

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
    evidence_score: float = Field(default=0.0, description="Average cross-encoder score of top contexts (0-1)")
    verification_score: Optional[float] = Field(default=None, description="Average NLI entailment score of generated claims (0-1)")
    retrieval_count: int = Field(default=0, description="Number of chunks retrieved after reranking")

# ---------- RETRIEVAL FUNCTIONS ----------
def generate_hypothesis(query: str) -> str:
    """Generate a hypothetical answer for HyDE (optional)."""
    prompt = f"Write a detailed answer to this question: {query}"
    inputs = hyde_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hyde_model.generate(**inputs, max_new_tokens=150)
    return hyde_tokenizer.decode(outputs[0], skip_special_tokens=True)

def retrieve_evidence(query: str) -> Tuple[List[dict], float]:
    """
    Retrieves evidence chunks and returns:
      - contexts: list of dicts with text, metadata, score, file_name
      - avg_top_score: average cross-encoder score of top 3 contexts (evidence strength)
    """
    expanded_query = expand_query(query)
    if HYDE_ENABLED:
        hyde_doc = generate_hypothesis(expanded_query)
        combined = expanded_query + " " + hyde_doc
    else:
        combined = expanded_query

    q_emb = embed_model.encode([combined], convert_to_tensor=True)
    q_emb = q_emb.cpu().numpy().astype('float32')
    # Retrieve more candidates than needed for reranking
    distances, indices = index.search(q_emb, TOP_K * 3)

    candidates = [texts[i] for i in indices[0]]
    pairs = [[query, doc] for doc in candidates]
    scores = cross_encoder.predict(pairs)

    # Sort by score descending
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
        if len(contexts) >= MAX_CHUNKS:
            break

    if contexts:
        # Use average of top 3 scores as evidence quality
        top_3 = contexts[:3]
        avg_top = sum(c["score"] for c in top_3) / len(top_3) if top_3 else 0.0
    else:
        avg_top = 0.0

    return contexts, avg_top

# ---------- PROMPT BUILDING ----------
def build_prompt(query: str, contexts: List[dict]) -> str:
    chunks = contexts[:MAX_CHUNKS]
    context_parts = []
    for i, c in enumerate(chunks):
        text = c["text"]
        if len(text) > CHUNK_TRUNCATE:
            # Truncate at sentence boundary if possible
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
        "You are a 3GPP standards expert. You MUST answer ONLY using the provided context.\n"
        "You are FORBIDDEN from using any external knowledge, even if you think you know the answer.\n\n"
        "STRICT RULES:\n"
        "1. If the user greets you (e.g., 'hi', 'hello'), respond warmly and explain your purpose.\n"
        "2. If the user asks about your capabilities, introduce yourself as a 3GPP chatbot.\n"
        "3. If the answer to the technical question is fully contained in the provided context, provide a clear, cited answer.\n"
        "4. If the context does NOT contain enough information to answer, respond EXACTLY: "
        "'I cannot find this information in the provided documents.'\n"
        "5. If the question is completely unrelated to 3GPP/telecom, respond: "
        "'I can only answer questions about 3GPP standards.'\n"
        "6. Every factual claim MUST include a citation like [Source: ...].\n"
        "7. Do NOT add any information that is not explicitly stated in the context.\n"
        "8. If the context is ambiguous, say so rather than guessing.\n\n"
        "--- PROVIDED CONTEXT ---\n"
        f"{context_text}\n"
        "--- END OF CONTEXT ---\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return system

# ---------- LLM GENERATION ----------
def generate_llm_response(prompt: str, stream: bool = False):
    if LLM_PROVIDER == "groq":
        return _generate_groq(prompt, stream)
    elif LLM_PROVIDER == "ollama":
        return _generate_ollama(prompt, stream)
    else:
        raise ValueError(f"Unknown provider: {LLM_PROVIDER}")

def _generate_groq(prompt: str, stream: bool):
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

# ---------- CLAIM VERIFICATION (NLI) ----------
def verify_claims(answer: str, contexts: List[dict], threshold: float) -> Tuple[bool, float, List[str]]:
    """
    Splits the answer into sentences and checks each against the provided contexts using NLI.
    Returns:
      - is_grounded: True if average entailment score >= threshold
      - avg_entailment: average entailment probability over all sentences
      - suspicious_sentences: list of sentences with entailment < threshold
    """
    if not ENABLE_NLI_VERIFICATION or nli_model is None:
        # Verification disabled – trust the LLM (not recommended for zero hallucination)
        return True, 1.0, []

    sentences = re.split(r'(?<=[.!?]) +', answer.strip())
    if not sentences:
        return False, 0.0, []

    context_texts = [c["text"] for c in contexts[:MAX_CHUNKS]]

    entail_scores = []
    suspicious = []
    for sent in sentences:
        if not sent.strip():
            continue
        max_entail = 0.0
        for ctx_text in context_texts:
            logits = nli_model.predict([(sent, ctx_text)])[0]
            probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
            entail_prob = probs[1] if len(probs) == 3 else probs  # adjust if different NLI model
            if entail_prob > max_entail:
                max_entail = entail_prob
        entail_scores.append(max_entail)
        if max_entail < threshold:
            suspicious.append(sent)

    avg_entail = sum(entail_scores) / len(entail_scores) if entail_scores else 0.0
    grounded = avg_entail >= threshold
    return grounded, avg_entail, suspicious

# ---------- API ENDPOINTS ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/tunnel")
async def get_tunnel():
    if TUNNEL_URL:
        return {"tunnel_url": TUNNEL_URL}
    else:
        if ENABLE_TUNNEL:
            return {"error": "Tunnel not yet ready or failed to start."}
        else:
            return {"error": "Tunnel is disabled. Set ENABLE_TUNNEL=true to use it."}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Step 1: Retrieve evidence
        contexts, evidence_score = retrieve_evidence(req.question)

        # Step 2: Evidence gate – if weak evidence, abstain
        if evidence_score < EVIDENCE_THRESHOLD:
            return QueryResponse(
                answer="I cannot answer this question reliably based on the provided documents.",
                sources=[],
                abstained=True,
                evidence_score=evidence_score,
                retrieval_count=len(contexts)
            )

        # Step 3: Build prompt and generate answer
        prompt = build_prompt(req.question, contexts)
        answer = generate_llm_response(prompt, stream=False)

        # Step 4: Check for LLM refusal (if model ignored instructions)
        refusal_phrases = [
            "cannot find this information",
            "not in the provided documents",
            "only answer questions about 3gpp",
            "outside the scope"
        ]
        llm_refused = any(p in answer.lower() for p in refusal_phrases)
        if llm_refused:
            return QueryResponse(
                answer=answer,
                sources=[],
                abstained=True,
                evidence_score=evidence_score,
                retrieval_count=len(contexts)
            )

        # Step 5: Claim verification (if enabled)
        verification_score = None
        if ENABLE_NLI_VERIFICATION:
            grounded, avg_entail, suspicious = verify_claims(answer, contexts, NLI_THRESHOLD)
            verification_score = avg_entail
            if not grounded:
                # Abstain entirely to ensure near-zero hallucination
                return QueryResponse(
                    answer="I could not verify the generated answer against the source documents.",
                    sources=[],
                    abstained=True,
                    evidence_score=evidence_score,
                    verification_score=verification_score,
                    retrieval_count=len(contexts)
                )

        # Step 6: Prepare sources for response
        sources = [
            Source(
                text=c["text"][:500],
                metadata=c["metadata"],
                score=c["score"],
                file_name=c.get("file_name", c["metadata"].get("file_name", "unknown"))
            )
            for c in contexts[:5]
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            abstained=False,
            evidence_score=evidence_score,
            verification_score=verification_score,
            retrieval_count=len(contexts)
        )

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
            # Retrieve evidence
            contexts, evidence_score = retrieve_evidence(req.question)

            # Evidence gate
            if evidence_score < EVIDENCE_THRESHOLD:
                yield f"data: {json.dumps({'error': 'Insufficient evidence to answer.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            prompt = build_prompt(req.question, contexts)
            completion = generate_llm_response(prompt, stream=True)

            # Stream tokens (note: claim verification is not performed in streaming mode)
            for token in completion:
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)