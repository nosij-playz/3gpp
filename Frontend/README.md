# Frontend – Flask Web Interface

A simple, responsive web UI for the 3GPP RAG chatbot.

---

## 📁 Contents

| File | Purpose |
|------|---------|
| `webapp.py` | Flask application |
| `templates/chat.html` | HTML + CSS + JS for the chat interface |

---

## 🚀 Running the Frontend

```bash
cd Frontend
pip install -r requirements.txt
python webapp.py --url http://localhost:8000
```

The interface runs at `http://localhost:5000`.

### Changing the Backend URL

- Use the **Settings** icon (⚙️) in the UI to change the backend URL at runtime.
- Or pass `--url` when starting the app.

---

## 🧩 Features

- Clean, responsive chat interface
- Real‑time message display
- Sources and abstention status shown
- Settings panel to change backend URL
- Keyboard shortcuts (Enter to send)

---

## 📦 Dependencies

- `Flask`
- `requests`
