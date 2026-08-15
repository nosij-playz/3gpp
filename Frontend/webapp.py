"""
Flask frontend for 3GPP RAG Chatbot.
Connects to the FastAPI backend (local or Cloudflare tunnel).
Usage: python webapp.py [--url http://localhost:8000]
"""

import argparse
import requests
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change for production

# Default backend URL (can be overridden by command line or web settings)
DEFAULT_BACKEND_URL = "http://localhost:8000"

# ---------- Command Line Argument ----------
parser = argparse.ArgumentParser()
parser.add_argument("--url", default=DEFAULT_BACKEND_URL, help="Backend API base URL")
args = parser.parse_args()

# Store the initial URL (will be used if session doesn't have one)
INITIAL_URL = args.url

# ---------- Routes ----------
@app.route('/', methods=['GET'])
def index():
    """Main chat page."""
    # If the backend URL is not in session, set it from command line.
    if 'backend_url' not in session:
        session['backend_url'] = INITIAL_URL
    return render_template('chat.html', backend_url=session['backend_url'])

@app.route('/settings', methods=['POST'])
def settings():
    """Update the backend URL from the settings form."""
    new_url = request.form.get('backend_url', '').strip()
    if new_url:
        # Remove trailing slash if any
        if new_url.endswith('/'):
            new_url = new_url[:-1]
        session['backend_url'] = new_url
        return jsonify({'status': 'ok', 'backend_url': new_url})
    else:
        return jsonify({'status': 'error', 'message': 'URL cannot be empty'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    """Send a question to the backend and return the answer."""
    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400

    backend_url = session.get('backend_url', DEFAULT_BACKEND_URL)
    # Ensure the backend URL is reachable
    try:
        # Quick health check (optional)
        health_resp = requests.get(f"{backend_url}/health", timeout=2)
        if health_resp.status_code != 200:
            return jsonify({'error': 'Backend is not healthy'}), 503
    except:
        return jsonify({'error': 'Cannot reach backend. Please check the URL.'}), 503

    # Send the query to the backend
    try:
        resp = requests.post(
            f"{backend_url}/query",
            json={'question': question, 'stream': False},
            timeout=120
        )
        if resp.status_code == 200:
            data = resp.json()
            return jsonify(data)
        else:
            return jsonify({'error': f"Backend error: {resp.status_code} - {resp.text}"}), 500
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timed out. The backend might be overloaded.'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True)