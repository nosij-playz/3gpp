# 🏗️ RAG Preparation – Knowledge Base Engineering

This module is responsible for the "Memory" part of the RAG system. It handles the end-to-end pipeline of converting raw 3GPP PDF specifications into a high-performance searchable vector index.

---

## 📁 Core Tooling

| File | Purpose |
|------|---------|
| `RAG_stepup.ipynb` | **The Indexing Pipeline**: A Jupyter notebook that implements the full ETL (Extract, Transform, Load) process. |

---

## 🛠️ The Indexing Pipeline (Detailed)

The `RAG_stepup.ipynb` notebook performs the following steps:

1. **Ingestion**: Downloads target 3GPP specifications (e.g., TS 38.300, TS 23.501) via API or local files.
2. **Parsing & Cleaning**: Extracts text from PDFs and removes noise (headers, footers, redundant page numbers).
3. **Semantic Chunking**: Splits documents into overlapping chunks to preserve context across boundaries.
4. **Embedding Generation**: Uses the `all-MiniLM-L6-v2` model to convert text chunks into 384-dimensional vectors.
5. **FAISS Indexing**: Builds a FlatL2 index for efficient similarity search.
6. **Serialization**: Saves the index and metadata (`nodes.pkl`, `texts.pkl`) for use by the Backend.

---

## 🚀 Usage Guide

### Option A: Use Pre-built Storage (Fastest)
If you don't need to change the documents, simply place these files in `Backend/storage/`:
- `faiss.index`
- `nodes.pkl`
- `texts.pkl`

### Option B: Build from Scratch
1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Pipeline**:
   Open `RAG_stepup.ipynb` in Jupyter and run all cells.
3. **Deploy**:
   Move the generated files from the notebook directory to `Backend/storage/`.

---

## 🔧 Advanced Customization

You can modify the pipeline to improve retrieval:
- **Expand Corpus**: Add new TS numbers to the download list.
- **Tune Chunking**: Adjust `chunk_size` and `chunk_overlap` to balance granularity and context.
- **Upgrade Embeddings**: Replace the embedding model with a larger one (e.g., `BAAI/bge-large-en-v1.5`) for better semantic capture.

---

## 📦 Output Artifacts

| Artifact | Description |
|----------|-------------|
| `faiss.index` | The binary vector space for fast nearest-neighbor search. |
| `nodes.pkl` | Metadata objects (filename, clause, page) mapped to vectors. |
| `texts.pkl` | The original text chunks used for LLM context. |
