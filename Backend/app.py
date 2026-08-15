"""
Terminal client for the 3GPP RAG API.
Usage: python cli.py [--url http://localhost:8000]
"""

import argparse
import json
import requests
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    print(f"🔗 Connecting to {base}")
    print("Type your question, or 'exit' to quit.\n")

    while True:
        try:
            query = input("> ").strip()
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if not query:
                continue

            print("\n⏳ Sending request...")
            resp = requests.post(
                f"{base}/query",
                json={"question": query, "stream": False},
                timeout=120
            )
            if resp.status_code != 200:
                print(f"❌ Server error: {resp.status_code} - {resp.text}")
                continue

            data = resp.json()
            print("\n" + "-"*60)
            print("💬 Answer:")
            print(data["answer"])
            print("-"*60)

            if data.get("sources"):
                print("📖 Sources:")
                for src in data["sources"][:3]:
                    print(f"  - {src['metadata'].get('file_name', 'unknown')} (score: {src['score']:.3f})")
            print("="*60 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()