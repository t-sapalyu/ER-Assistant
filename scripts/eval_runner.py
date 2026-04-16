import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import client, load_index, retrieve, generate_answer
from cost_tracker import track_cost

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ---------------------------
# LLM-as-Judge
# ---------------------------
def judge_faithfulness(question, answer, context):
    """Ask gpt-4o-mini to score whether the answer is faithful to the context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge. Given a question, an answer, "
                    "and the context that was provided, determine if the answer "
                    "is faithful to the context. Score: 'faithful' if every claim "
                    "in the answer is supported by the context, 'unfaithful' if "
                    "the answer contains information not in the context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Context: {context}\n\n"
                    f"Answer: {answer}\n\n"
                    "Is the answer faithful to the context? Reply with 'faithful' or 'unfaithful' and one sentence explaining why."
                ),
            },
        ],
        temperature=0,
        max_tokens=100,
    )
    track_cost(response)
    return response.choices[0].message.content.strip()


# ---------------------------
# Main eval runner
# ---------------------------
def run_eval():
    print("Loading index...")
    index, chunks = load_index()

    with open(DATA_DIR / "questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    faithful_count = 0

    print(f"Running {len(questions)} questions...\n")

    for q in questions:
        question = q["question"]
        expected = q.get("expected_answer", "")
        category = q.get("category", "unknown")

        # Retrieve + generate
        retrieved = retrieve(question, index, chunks, k=5)
        answer = generate_answer(question, retrieved)

        # Build context string for the judge
        context = "\n\n".join(c["text"] for c in retrieved)

        # Judge
        verdict = judge_faithfulness(question, answer, context)
        is_faithful = "unfaithful" not in verdict.lower()
        if is_faithful:
            faithful_count += 1

        status = "FAITHFUL  ✅" if is_faithful else "UNFAITHFUL ❌"
        print(f"[{status}] [{category}] {question[:70]}")

        results.append(
            {
                "question": question,
                "category": category,
                "expected": expected,
                "answer": answer,
                "verdict": verdict,
                "faithful": is_faithful,
                "sources": [
                    {"source": c["source"], "page": c["page"], "score": c["score"]}
                    for c in retrieved
                ],
            }
        )

    # Summary
    total = len(results)
    print(f"\n{'='*60}")
    print(f"RESULT: {faithful_count}/{total} answers faithful ({faithful_count/total*100:.0f}%)")

    # Breakdown by category
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, {"total": 0, "faithful": 0})
        categories[cat]["total"] += 1
        if r["faithful"]:
            categories[cat]["faithful"] += 1

    print("\nBreakdown by category:")
    for cat, counts in categories.items():
        pct = counts["faithful"] / counts["total"] * 100
        print(f"  {cat}: {counts['faithful']}/{counts['total']} ({pct:.0f}%)")

    # Save to disk
    out_path = DATA_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    run_eval()



"""if __name__ == "__main__":
    # run_eval()  # comment this out

    # Test a single question manually
    index, chunks = load_index()
    q = "What should you do if someone is choking?"
    retrieved = retrieve(q, index, chunks, k=5)
    answer = generate_answer(q, retrieved)
    context = "\n\n".join(c["text"] for c in retrieved)
    verdict = judge_faithfulness(q, answer, context)

    print(f"Q: {q}")
    print(f"A: {answer}")
    print(f"Verdict: {verdict}")"""
