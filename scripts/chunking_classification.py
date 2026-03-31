import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from chunking_common import (
    MAX_CHUNK_TOKENS,
    MODELS,
    chunk_by_sentences_tokens,
    chunk_fixed_tokens,
    load_book,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "results_chunking_classification.json"

TEST_SIZE = 0.3
RANDOM_STATE = 42
RF_N_ESTIMATORS = 100


def evaluate_classification(embeddings, labels):
    y = np.asarray(labels, dtype=np.int64)
    stratified = True
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    except ValueError:
        stratified = False
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=None,
        )

    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    return {
        "accuracy": round(acc, 4),
        "train_chunks": int(len(y_train)),
        "test_chunks": int(len(y_test)),
        "stratified": stratified,
    }


def main():
    chapters = load_book()
    results = []

    for model_name, model_id in MODELS:
        model = SentenceTransformer(model_id)
        tok = model.tokenizer

        fixed_chunks, fixed_labels = chunk_fixed_tokens(chapters, tok)
        sent_chunks, sent_labels = chunk_by_sentences_tokens(chapters, tok)

        emb_f = model.encode(fixed_chunks, show_progress_bar=False)
        emb_s = model.encode(sent_chunks, show_progress_bar=False)

        fm = evaluate_classification(np.asarray(emb_f), fixed_labels)
        sm = evaluate_classification(np.asarray(emb_s), sent_labels)

        results.append(
            {
                "model": model_name,
                "max_tokens_per_chunk": MAX_CHUNK_TOKENS,
                "fixed_chunk_count": len(fixed_chunks),
                "sentence_chunk_count": len(sent_chunks),
                "fixed_accuracy": fm["accuracy"],
                "sentence_accuracy": sm["accuracy"],
                "fixed_split": {
                    "train": fm["train_chunks"],
                    "test": fm["test_chunks"],
                    "stratified": fm["stratified"],
                },
                "sentence_split": {
                    "train": sm["train_chunks"],
                    "test": sm["test_chunks"],
                    "stratified": sm["stratified"],
                },
            }
        )

    OUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
