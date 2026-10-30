import json, os
from pathlib import Path

# Project-root file, regardless of the directory the script is run from
COST_FILE = Path(__file__).resolve().parent.parent / "cost_tracker.json"

def track_cost(response, is_embedding=False):
    usage = response.usage
    if is_embedding:
        cost = usage.total_tokens * 0.02 / 1_000_000
    else:  # chat (gpt-4o-mini)
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000
    if os.path.exists(COST_FILE):
        with open(COST_FILE) as f:
            data = json.load(f)
    else:
        data = {"total": 0.0}
    data["total"] += cost
    with open(COST_FILE, "w") as f:
        json.dump(data, f)
    print(f"This call: ${cost:.6f} | Team total: ${data['total']:.4f} / $5.00")