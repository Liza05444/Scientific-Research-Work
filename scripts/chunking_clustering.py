import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from chunking_common import (
    MAX_CHUNK_TOKENS,
    MODELS,
    chunk_by_sentences_tokens,
    chunk_fixed_tokens,
    load_book,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "results_chunking_clustering.json"


def main():
    chapters = load_book()
    k = len(chapters)
    results = []

    for model_name, model_id in MODELS:
        model = SentenceTransformer(model_id)
        tok = model.tokenizer

        fixed_chunks, fixed_labels = chunk_fixed_tokens(chapters, tok)
        sent_chunks, sent_labels = chunk_by_sentences_tokens(chapters, tok)

        emb_f = model.encode(fixed_chunks, show_progress_bar=False)
        emb_s = model.encode(sent_chunks, show_progress_bar=False)

        cf = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(emb_f)
        cs = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(emb_s)

        results.append(
            {
                "model": model_name,
                "max_tokens_per_chunk": MAX_CHUNK_TOKENS,
                "fixed_chunk_count": len(fixed_chunks),
                "sentence_chunk_count": len(sent_chunks),
                "chunking_fixed_ARI": round(
                    adjusted_rand_score(fixed_labels, cf), 4
                ),
                "chunking_fixed_NMI": round(
                    normalized_mutual_info_score(fixed_labels, cf), 4
                ),
                "chunking_by_sentences_ARI": round(
                    adjusted_rand_score(sent_labels, cs), 4
                ),
                "chunking_by_sentences_NMI": round(
                    normalized_mutual_info_score(sent_labels, cs), 4
                ),
            }
        )

    OUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
