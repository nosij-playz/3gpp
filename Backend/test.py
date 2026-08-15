"""
Automated test suite for 3GPP RAG API.
Runs a set of predefined queries and saves results to output.txt.
Usage: python test.py [--url http://localhost:8000]
"""

import argparse
import json
import requests
import time
import sys
from datetime import datetime

# ---------- CONFIG ----------
DEFAULT_URL = "http://localhost:8000"
OUTPUT_FILE = "Report.txt"

# ---------- QUERY SUITE ----------
QUERIES = [
    # ===== In-Scope Technical Questions (Should answer with citations) =====
    {
        "id": "t1",
        "category": "Technical",
        "question": "What is the role of the AMF in 5G?"
    },
    {
        "id": "t2",
        "category": "Technical",
        "question": "Explain the 5G QoS flow mapping."
    },
    {
        "id": "t3",
        "category": "Technical",
        "question": "What are the main functions of the gNB?"
    },
    {
        "id": "t4",
        "category": "Technical",
        "question": "What is the N2 interface?"
    },
    {
        "id": "t5",
        "category": "Technical",
        "question": "Describe the 5G registration procedure."
    },
    {
        "id": "t6",
        "category": "Technical",
        "question": "What security features does 5G provide?"
    },
    {
        "id": "t7",
        "category": "Technical",
        "question": "What is 5G‑NR?"
    },

    # ===== Abbreviations & Short Forms =====
    {
        "id": "e1",
        "category": "Edge",
        "question": "What is AMF?"
    },
    {
        "id": "e2",
        "category": "Edge",
        "question": "Define NR."
    },

    # ===== Broad / Open-Ended =====
    {
        "id": "e3",
        "category": "Edge",
        "question": "Tell me everything about 5G security."
    },
    {
        "id": "e4",
        "category": "Edge",
        "question": "Summarize the 5G architecture."
    },

    # ===== Specific / Detailed =====
    {
        "id": "e5",
        "category": "Edge",
        "question": "What is the difference between QoS and QoE in 5G?"
    },
    {
        "id": "e6",
        "category": "Edge",
        "question": "Explain the 5G authentication procedure step by step."
    },

    # ===== Out-of-Scope (Should Refuse) =====
    {
        "id": "o1",
        "category": "Out-of-Scope",
        "question": "Who is the CEO of Nokia?"
    },
    {
        "id": "o2",
        "category": "Out-of-Scope",
        "question": "How much does a 5G base station cost?"
    },
    {
        "id": "o3",
        "category": "Out-of-Scope",
        "question": "What is the weather today?"
    },
    {
        "id": "o4",
        "category": "Out-of-Scope",
        "question": "Explain the meaning of life."
    },

    # ===== Edge Cases & Stress Tests =====
    {
        "id": "s1",
        "category": "Stress",
        "question": "What is 5G?"   # very broad
    },
    {
        "id": "s2",
        "category": "Stress",
        "question": "Tell me about the AMF, SMF, and UPF."  # multiple entities
    },
    {
        "id": "s3",
        "category": "Stress",
        "question": "What is the role of the AMF in 5G? Also explain how it interacts with the gNB."  # compound
    },
    {
        "id": "s4",
        "category": "Stress",
        "question": "Explain 5G QoS in one sentence."  # expects concise
    },
    {
        "id": "s5",
        "category": "Stress",
        "question": "What are the frequency bands for NR?"  # may not be in docs
    },
    {
        "id": "s6",
        "category": "Stress",
        "question": "What is the difference between 5G NSA and SA?"  # architecture
    },
    {
        "id": "s7",
        "category": "Stress",
        "question": "What is the maximum data rate of 5G?"  # may not be explicit
    },
    {
        "id": "s8",
        "category": "Stress",
        "question": "How does beamforming work in 5G?"  # may be in docs
    },
    {
        "id": "s9",
        "category": "Stress",
        "question": ""  # empty input
    },
    {
        "id": "s10",
        "category": "Stress",
        "question": "What is the 3GPP release 18?"  # version specific
    }
]

# ---------- TEST RUNNER ----------
def run_tests(base_url):
    results = []

    print(f"\n🚀 Starting test suite against {base_url}")
    print(f"   Total queries: {len(QUERIES)}\n")

    for q in QUERIES:
        query_text = q["question"]
        if not query_text.strip():
            print(f"   ⏭️  Skipping empty query (id: {q['id']})")
            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": "(empty)",
                "status": "skipped",
                "answer": None,
                "sources": None,
                "error": "Empty query"
            })
            continue

        print(f"   ➤ [{q['id']}] {query_text[:50]}...", end="", flush=True)

        try:
            start = time.time()
            resp = requests.post(
                f"{base_url}/query",
                json={"question": query_text, "stream": False},
                timeout=120
            )
            elapsed = time.time() - start

            if resp.status_code != 200:
                print(f" ❌ Error {resp.status_code}")
                results.append({
                    "id": q["id"],
                    "category": q["category"],
                    "question": query_text,
                    "status": "error",
                    "answer": None,
                    "sources": None,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}"
                })
                continue

            data = resp.json()
            print(f" ✅ {elapsed:.2f}s")

            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": query_text,
                "status": "success",
                "answer": data.get("answer"),
                "sources": data.get("sources", [])[:3],  # top 3
                "error": None,
                "elapsed": elapsed
            })

        except requests.exceptions.Timeout:
            print(" ⏰ Timeout")
            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": query_text,
                "status": "timeout",
                "answer": None,
                "sources": None,
                "error": "Request timed out after 120s"
            })
        except Exception as e:
            print(f" ❌ Exception: {e}")
            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": query_text,
                "status": "exception",
                "answer": None,
                "sources": None,
                "error": str(e)
            })

    return results

# ---------- OUTPUT SAVER ----------
def save_results(results, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("3GPP RAG API Test Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Summary stats
        total = len(results)
        success = sum(1 for r in results if r["status"] == "success")
        errors = sum(1 for r in results if r["status"] in ("error", "exception", "timeout"))
        f.write(f"Total queries: {total}\n")
        f.write(f"Successful:    {success}\n")
        f.write(f"Errors:        {errors}\n")
        f.write(f"Skipped:       {total - success - errors}\n")
        f.write("\n" + "-" * 80 + "\n\n")

        # Detailed results
        for i, r in enumerate(results, 1):
            f.write(f"[{i}] ID: {r['id']} | Category: {r['category']}\n")
            f.write(f"    Query: {r['question']}\n")
            f.write(f"    Status: {r['status']}\n")
            if r["status"] == "success":
                f.write(f"    Elapsed: {r.get('elapsed', 0):.2f}s\n")
                f.write("    Answer:\n")
                # Indent answer
                answer_lines = r["answer"].splitlines() if r["answer"] else []
                for line in answer_lines:
                    f.write(f"        {line}\n")
                f.write("    Sources:\n")
                for src in r.get("sources", []):
                    name = src.get("metadata", {}).get("file_name", "unknown")
                    score = src.get("score", 0)
                    f.write(f"        - {name} (score: {score:.3f})\n")
            elif r["status"] == "skipped":
                f.write("    Skipped (empty query)\n")
            else:
                f.write(f"    Error: {r.get('error', 'Unknown error')}\n")
            f.write("\n" + "-" * 40 + "\n\n")

    print(f"\n✅ Results saved to {filename}")

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"🔗 Connecting to {base_url}")

    # Check health first
    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        if health.status_code != 200:
            print(f"❌ Server not healthy: {health.status_code}")
            sys.exit(1)
        print("✅ Server is healthy\n")
    except Exception as e:
        print(f"❌ Cannot reach server: {e}")
        sys.exit(1)

    results = run_tests(base_url)
    save_results(results, OUTPUT_FILE)

if __name__ == "__main__":
    main()