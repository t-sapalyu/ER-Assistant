# app/rag_api.py
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from flask import Flask, request, jsonify
from rag_pipeline import load_index, ask

app = Flask(__name__)

print("Loading FAISS index...")
index, chunks = load_index()
print(f"Ready. {index.ntotal} vectors loaded.")

@app.route('/query', methods=['POST'])
def query():
    data     = request.json
    question = data.get('question', '').strip()

    if not question:
        return jsonify({"answer": "Please ask a question."}), 400

    print(f"[WhatsApp] Question: {question}")

    result = ask(question, index, chunks)
    answer = result["answer"] if isinstance(result, dict) else str(result)

    print(f"[WhatsApp] Answer: {answer[:100]}...")
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(port=5000, debug=False)