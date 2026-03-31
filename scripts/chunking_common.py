import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK_PATH = ROOT / "data" / "book.json"

MODELS = [
    ("e5-large", "intfloat/multilingual-e5-large"),
    ("bge-m3", "BAAI/bge-m3"),
    ("e5-small", "intfloat/multilingual-e5-small"),
]

MAX_CHUNK_TOKENS = 508


def load_book():
    return json.loads(BOOK_PATH.read_text(encoding="utf-8"))["chapters"]


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _decode(tokenizer, ids):
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _encode_ids(tokenizer, text):
    prev = getattr(tokenizer, "model_max_length", 512)
    tokenizer.model_max_length = 1_000_000
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    finally:
        tokenizer.model_max_length = prev


def _append_chunk_if_nonempty(tokenizer, ids, chunks, labels, ch_idx):
    t = _decode(tokenizer, ids)
    if t:
        chunks.append(t)
        labels.append(ch_idx)


def chunk_fixed_tokens(chapters, tokenizer):
    chunks, labels = [], []
    n = MAX_CHUNK_TOKENS
    for ch_idx, ch in enumerate(chapters):
        ids = _encode_ids(tokenizer, ch["text"])
        for i in range(0, len(ids), n):
            _append_chunk_if_nonempty(tokenizer, ids[i : i + n], chunks, labels, ch_idx)
    return chunks, labels


def chunk_by_sentences_tokens(chapters, tokenizer):
    chunks, labels = [], []
    n = MAX_CHUNK_TOKENS

    for ch_idx, ch in enumerate(chapters):
        cur = []
        for sent in _split_sentences(ch["text"]):
            sid = _encode_ids(tokenizer, sent)
            if len(sid) > n:
                if cur:
                    _append_chunk_if_nonempty(tokenizer, cur, chunks, labels, ch_idx)
                    cur = []
                for j in range(0, len(sid), n):
                    _append_chunk_if_nonempty(tokenizer, sid[j : j + n], chunks, labels, ch_idx)
                continue
            if cur and len(cur) + len(sid) > n:
                _append_chunk_if_nonempty(tokenizer, cur, chunks, labels, ch_idx)
                cur = []
            cur.extend(sid)
        if cur:
            _append_chunk_if_nonempty(tokenizer, cur, chunks, labels, ch_idx)

    return chunks, labels
