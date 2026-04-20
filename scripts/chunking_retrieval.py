import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from chunking_common import MODELS, load_book

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "questions.json"
OUT_PATH = ROOT / "results" / "results_chunking_retrieval.json"

CHUNK_SIZES = [128, 256, 512]
OVERLAP_PCTS = [0, 5, 10, 15, 20, 25]
CONFIGS = [(n, p, round(n * p / 100)) for n in CHUNK_SIZES for p in OVERLAP_PCTS]
TOP_K_PRECISION = 1
TOP_K_RECALL = 5


def chunk_with_overlap(chapters, tokenizer, n, overlap):
    chunks, spans = [], []
    step = max(1, n - overlap)
    prev = getattr(tokenizer, "model_max_length", 512)
    tokenizer.model_max_length = 1_000_000
    try:
        for ch_idx, ch in enumerate(chapters):
            text = ch["text"]
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            off = enc["offset_mapping"]
            i = 0
            while i < len(off):
                j = min(i + n, len(off))
                s, e = off[i][0], off[j - 1][1]
                piece = text[s:e].strip()
                if piece:
                    chunks.append(piece)
                    spans.append((ch_idx, s, e))
                if j >= len(off):
                    break
                i += step
    finally:
        tokenizer.model_max_length = prev
    return chunks, spans


def main():
    chapters = load_book()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    q_texts = [q["question"] for q in questions]
    targets = []
    for q in questions:
        ch_i = q["chapter"] - 1
        pos = chapters[ch_i]["text"].find(q["source"])
        targets.append((ch_i, pos, pos + len(q["source"])))

    results = []
    for model_name, model_id in MODELS:
        model = SentenceTransformer(model_id)
        tok = model.tokenizer
        q_emb = np.asarray(model.encode(q_texts, show_progress_bar=False, normalize_embeddings=True))

        for n, pct, ov in CONFIGS:
            chunks, spans = chunk_with_overlap(chapters, tok, n, ov)
            c_emb = np.asarray(model.encode(chunks, show_progress_bar=False, normalize_embeddings=True))
            sims = q_emb @ c_emb.T
            order = np.argsort(-sims, axis=1)
            top_p = order[:, :TOP_K_PRECISION]
            top_r = order[:, :TOP_K_RECALL]

            precisions, recalls = [], []
            for qi, (ch_i, pos, end) in enumerate(targets):
                relevant = [
                    sp_ch == ch_i and s <= pos and end <= e
                    for sp_ch, s, e in spans
                ]
                total_rel = sum(relevant)
                tp_p = sum(1 for idx in top_p[qi] if relevant[idx])
                tp_r = sum(1 for idx in top_r[qi] if relevant[idx])
                precisions.append(tp_p / TOP_K_PRECISION)
                recalls.append(tp_r / total_rel if total_rel else 0.0)

            results.append(
                {
                    "model": model_name,
                    "chunk_size": n,
                    "overlap_pct": pct,
                    "overlap": ov,
                    "chunk_count": len(chunks),
                    "precision_top_k": TOP_K_PRECISION,
                    "recall_top_k": TOP_K_RECALL,
                    "precision": round(float(np.mean(precisions)), 4),
                    "recall": round(float(np.mean(recalls)), 4),
                }
            )

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
