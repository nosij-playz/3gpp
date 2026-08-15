# RAG Preparation – Building the Knowledge Base

This folder contains the Jupyter notebook that downloads 3GPP specifications and builds the FAISS index.

---

## 📁 Contents

| File | Purpose |
|------|---------|
| `RAG_stepup.ipynb` | Notebook that downloads specs, chunks, embeds, and saves the index. |

---

## 🚀 Running the Notebook

### Option A – Use Pre‑built Storage (Recommended)

Download the following files from the provided link and place them in `Backend/storage/`:
- `faiss.index`
- `nodes.pkl`
- `texts.pkl`

### Option B – Build from Scratch

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open the notebook:
   ```bash
   jupyter notebook RAG_stepup.ipynb
   ```

3. Run all cells. The notebook will:
   - Download 3GPP specs (TS 38.300, TS 23.501, TS 33.501).
   - Extract and chunk the documents.
   - Generate embeddings using `all-MiniLM-L6-v2`.
   - Build the FAISS index.
   - Save `faiss.index`, `nodes.pkl`, `texts.pkl` to the `storage/` folder.

---

## 🔧 Customisation

You can modify the notebook to:
- Add more specifications (e.g., TS 23.502, TS 38.410).
- Change chunk size / overlap.
- Use a different embedding model (e.g., `BAAI/bge-large-en-v1.5`).

---

## 📂 Output Files

| File | Description |
|------|-------------|
| `faiss.index` | Vector index for similarity search |
| `nodes.pkl` | LlamaIndex node objects with metadata |
| `texts.pkl` | Plain text chunks for retrieval |

**These three files must be placed in `Backend/storage/` for the server to work.**
