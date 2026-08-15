# Backend – FastAPI Server & CLI

This folder contains the core RAG backend: FastAPI server, CLI client, and test suite.

---

## 📁 Contents

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app with evidence‑controlled RAG |
| `cli.py` | Terminal‑based interactive client |
| `test.py` | Automated test suite |
| `config.py` | All settings (loaded from `.env`) |
| `storage/` | Pre‑built FAISS index and pickle files |
| `.env` | Environment variables (API keys, model names) |

---

## 🚀 Running the Server

```bash
cd Backend
pip install -r requirements.txt
cp .env.example .env   # edit with your keys
python server.py
```

Server runs at `http://localhost:8000`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Ask a question (non‑streaming) |
| `/query/stream` | POST | Ask a question (streaming) |
| `/tunnel` | GET | Get Cloudflare tunnel URL (if enabled) |

### Example `curl`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the role of the AMF?"}'
```

---

## 🧪 Testing

```bash
python test.py --url http://localhost:8000
```

Generates `Report.txt` with detailed results.

---

## 🔧 Configuration

See `.env` for all configurable options.

### Provider Options

- **Groq** – fast, hosted (requires `GROQ_API_KEY`)
- **Ollama** – local, offline (requires `ollama serve` running)

---

## 📦 Dependencies

See `requirements.txt`. Key packages:
- `fastapi`, `uvicorn`
- `sentence-transformers`, `faiss-cpu`
- `groq`, `transformers`
- `numpy`, `requests`
