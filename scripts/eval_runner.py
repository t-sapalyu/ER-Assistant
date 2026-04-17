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
JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for a first-aid RAG assistant.
Given a question, the retrieved context, and the system's answer, classify the answer
into EXACTLY one of four verdicts:

  faithful        — the answer is directly supported by the context and addresses the question
  hallucinated    — the answer contains claims that are NOT present in the context
  missed          — the context clearly contains enough information to answer the question,
                    but the answer incorrectly says "I don't have enough information"
  correct_refusal — the context does not contain relevant information and the answer
                    correctly declines to answer

Reply with one word (the verdict), then a colon, then one sentence explaining why.
Example: "missed: The context on page 51 describes chest thrusts for pregnant persons but the answer ignored it."
"""


def judge_faithfulness(question, answer, context, top_score: float = 1.0):
    """Classify answer as faithful / hallucinated / missed / correct_refusal."""
    # Fast-path: very low retrieval score + refusal → almost certainly correct_refusal
    refusal = "don't have enough" in answer.lower() or "i don't know" in answer.lower()
    if refusal and top_score < 0.35:
        return "correct_refusal: retrieval score too low to expect a grounded answer."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    f"Answer: {answer}\n\n"
                    "Verdict:"
                ),
            },
        ],
        temperature=0,
        max_tokens=120,
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

    VERDICT_PASS = {"faithful", "correct_refusal"}
    VERDICT_ICONS = {
        "faithful":        "✅ faithful       ",
        "correct_refusal": "✅ correct_refusal",
        "missed":          "⚠️  missed         ",
        "hallucinated":    "❌ hallucinated   ",
    }

    results = []
    verdict_counts = {"faithful": 0, "correct_refusal": 0, "missed": 0, "hallucinated": 0}

    print(f"Running {len(questions)} questions...\n")

    for q in questions:
        question = q["question"]
        expected = q.get("expected_answer", "")
        category = q.get("category", "unknown")

        # Retrieve + generate
        retrieved = retrieve(question, index, chunks, k=5)
        top_score = retrieved[0]["score"] if retrieved else 0.0
        answer = generate_answer(question, retrieved)

        # Build context string for the judge
        context = "\n\n".join(c["text"] for c in retrieved)

        # Judge — now 4-way verdict
        verdict = judge_faithfulness(question, answer, context, top_score)
        verdict_label = verdict.split(":")[0].strip().lower()
        if verdict_label not in verdict_counts:
            verdict_label = "hallucinated"   # safe fallback for unexpected output
        verdict_counts[verdict_label] += 1
        passed = verdict_label in VERDICT_PASS

        icon = VERDICT_ICONS.get(verdict_label, "❓")
        print(f"[{icon}] [{category}] {question[:70]}")

        results.append(
            {
                "question": question,
                "category": category,
                "expected": expected,
                "answer": answer,
                "verdict": verdict,
                "verdict_label": verdict_label,
                "passed": passed,
                "top_score": round(top_score, 4),
                "sources": [
                    {"source": c["source"], "page": c["page"], "score": c["score"]}
                    for c in retrieved
                ],
            }
        )

    # Summary
    total = len(results)
    passed_total = sum(1 for r in results if r["passed"])
    print(f"\n{'='*60}")
    print(f"PASSED: {passed_total}/{total} ({passed_total/total*100:.0f}%)")
    print(f"  ✅ faithful:         {verdict_counts['faithful']}")
    print(f"  ✅ correct_refusal:  {verdict_counts['correct_refusal']}")
    print(f"  ⚠️  missed:           {verdict_counts['missed']}")
    print(f"  ❌ hallucinated:     {verdict_counts['hallucinated']}")

    # Breakdown by category
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, {"total": 0, "passed": 0, "missed": 0, "hallucinated": 0})
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1
        elif r["verdict_label"] == "missed":
            categories[cat]["missed"] += 1
        else:
            categories[cat]["hallucinated"] += 1

    print("\nBreakdown by category:")
    for cat, counts in categories.items():
        pct = counts["passed"] / counts["total"] * 100
        print(f"  {cat}: {counts['passed']}/{counts['total']} passed ({pct:.0f}%) "
              f"| missed={counts['missed']} hallucinated={counts['hallucinated']}")

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

