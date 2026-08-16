# 🎨 Frontend – User Interface

The Frontend provides a clean, intuitive web interface for interacting with the 3GPP RAG system. It is built with Flask and focuses on transparency—showing the user not just the answer, but the grounding status.

---

## 📁 Components

| File | Purpose |
|------|---------|
| `webapp.py` | **Flask Server**: Handles routing and acts as a bridge between the UI and the FastAPI Backend. |
| `templates/chat.html` | **The View**: A responsive HTML/CSS/JS interface with a modern chat experience. |

---

## 🚀 Launching the UI

### 1. Installation
```bash
cd Frontend
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python webapp.py --url http://localhost:8000
```
The interface is available at `http://localhost:5000`.

---

## 🖼️ Screenshot

![Chat Interface](screenshot.png)
*(Add a screenshot of the chat UI here)*

---

## ✨ Key Features

- **Grounded Chat**: Real-time display of LLM responses.
- **Verification Transparency**: Clearly labels answers as "Grounded" or "Unverified" based on the Backend's NLI check.
- **Source Attribution**: Displays the 3GPP specifications and clauses used to generate the answer.
- **Dynamic Backend Config**: Change the API endpoint via the **Settings** (⚙️) panel without restarting the server.
- **Responsive Design**: Works across desktop and mobile browsers.

---

## 🔗 Connecting to the Backend

The frontend communicates with the FastAPI backend via HTTP. You can:
- Use the **local backend**: `http://localhost:8000`
- Use a **Cloudflare tunnel**: `https://xxxx.trycloudflare.com`
- Enter the URL manually in the **Settings** panel (⚙️)

**Note:** The backend must be running **before** starting the frontend.

---

## 🛠️ Technical Flow

`User Input` $\rightarrow$ `Flask App` $\rightarrow$ `FastAPI Backend` $\rightarrow$ `NLI Result` $\rightarrow$ `UI Rendering`

The UI renders different styles based on the `abstained` and `verification_score` flags returned by the API, providing immediate visual feedback on the reliability of the information.
