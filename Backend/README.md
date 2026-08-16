# ⚙️ Backend – FastAPI Server & Verification Engine

The Backend is the heart of the 3GPP RAG system. It orchestrates the retrieval pipeline, manages the LLM connection, and implements the critical NLI verification layer to ensure zero hallucinations.

---

## 📁 Core Components

| File | Purpose | Technical Detail |
|------|---------|-------------------|
| `server.py` | **Main Orchestrator** | Implements the `Query` $\rightarrow$ `Retrieval` $\rightarrow$ `NLI` pipeline. |
| `config.py` | **Central Settings** | Loads `.env` variables into Python objects for global access. |
| `cli.py` | **Interactive Client** | A lightweight terminal tool for rapid testing. |
| `test.py` | **Evaluation Suite** | Runs a battery of queries and generates `Report.txt`. |
| `storage/` | **Knowledge Base** | Stores the FAISS vector index and metadata pickle files. |
| `.env` | **Environment** | API keys, thresholds, and model selection. |

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd Backend
pip install -r requirements.txt
cp .env.example .env   # Edit with your API keys & model preferences
```

### 2. Launch Server
```bash
python server.py
```
Server runs at `http://localhost:8000`.

---

## 🛠️ API Reference

| Endpoint | Method | Description | Payload Example |
|----------|--------|-------------|------------------|
| `/health` | GET | System heartbeat | N/A |
| `/query` | POST | Standard grounded query | `{"question": "What is the AMF?"}` |
| `/query/stream` | POST | Streaming grounded query | `{"question": "...", "stream": true}` |
| `/tunnel` | GET | Get public Cloudflare URL | N/A |

### Example Request (curl)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the role of the AMF?"}'
```

---

## �️ Hallucination Prevention Pipeline

1. **Evidence Gate**: Before calling the LLM, the system checks if the average cross‑encoder score of the top 3 retrieved chunks exceeds `EVIDENCE_THRESHOLD`. If not, it abstains immediately.

2. **Controlled Generation**: The LLM is instructed to answer **only** from the provided context, never from its training data.

3. **NLI Verification**: The generated answer is split into sentences. Each sentence is checked against the retrieved context using a Cross‑Encoder NLI model. The answer is accepted if:
   - ≥70% of the claims are entailed **or**
   - The average entailment score is ≥80% of `NLI_THRESHOLD`

4. **Short‑answer bypass**: Answers under 150 characters are automatically accepted (prevents false positives for concise replies).

---

## 🐞 Troubleshooting

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| `500 Internal Server Error` | Model decommissioned or API key invalid | Update `GROQ_MODEL` in `.env` |
| Slow responses | HyDE enabled or high `MAX_CHUNKS` | Set `HYDE_ENABLED=false` or reduce `MAX_CHUNKS` |
| `"I cannot find this information"` | Evidence score below threshold | Lower `EVIDENCE_THRESHOLD` in `.env` |
| `"I could not verify"` | NLI score below threshold | Lower `NLI_THRESHOLD` in `.env` |
| CUDA out of memory | NLI model too large | Set `ENABLE_NLI_VERIFICATION=false` |

---

## 🧪 Evaluation & Testing

To verify system accuracy and hallucination rates:
```bash
python test.py --url http://localhost:8000
```
This generates a `Report.txt` containing:
- **Query**: The question asked.
- **Answer**: The LLM's response.
- **Status**: Grounded, Hallucination Risk, or Abstention.
- **Scores**: Retrieval and NLI scores.

---

## 📦 Dependencies
Key libraries: `fastapi`, `uvicorn`, `sentence-transformers`, `faiss-cpu`, `groq`, `transformers`.
