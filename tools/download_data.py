import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "data" / "sberquad.json"

ds = load_dataset("kuznetsoffandrey/sberquad", "sberquad", split="validation")

contexts = []
queries = []

for i, row in enumerate(ds):
    contexts.append(row["context"])
    queries.append(
        {
            "question": row["question"],
            "correctParagraphKey": i,
        }
    )

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump({"contexts": contexts, "queries": queries}, f, ensure_ascii=False)
