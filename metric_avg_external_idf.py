import argparse
import json
import math
import gzip
import re
import statistics
from pathlib import Path

# Simple tokenizer
TOKEN_RE = re.compile(r"[A-Za-z]+")

# Basic English stopwords (extend if needed)
STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","that","this","these","those","is","am","are","was","were","be","been",
    "being","to","of","in","on","for","with","as","by","at","from","it","its","not","no","do","does","did","so","such","can",
    "could","should","would","will","shall","may","might","must","have","has","had","i","you","he","she","we","they","them",
    "him","her","my","your","our","their","me","us","which","who","whom","what","when","where","why","how"
}

def load_model(path: Path):
    """Load IDF model built by build_idf_from_hf.py"""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as g:
            m = json.load(g)
    else:
        m = json.loads(path.read_text(encoding="utf-8"))
    N = int(m["N"])
    df = {k: int(v) for k, v in m["df"].items()}
    return N, df

def tokenize(text, keep_stopwords=False, min_len=2):
    """Tokenize plain text into lowercase words, filter short tokens & stopwords."""
    for m in TOKEN_RE.finditer(text):
        t = m.group(0).lower()
        if len(t) < min_len:
            continue
        if not keep_stopwords and t in STOPWORDS:
            continue
        yield t

def idf(N, df_count):
    """Compute smoothed IDF = log((N+1)/(df+1))"""
    return math.log((N + 1) / (df_count + 1))

def main():
    ap = argparse.ArgumentParser(description="Compute Average External IDF for a corpus.")
    ap.add_argument("corpus_path", type=Path, help="Path to corpus text file.")
    ap.add_argument("--idf-model", type=Path, required=True, help="Path to external IDF model (.json or .json.gz).")
    ap.add_argument("--keep-stopwords", action="store_true", help="Include stopwords in calculation.")
    ap.add_argument("--min-len", type=int, default=2, help="Minimum token length (default=2).")
    ap.add_argument("--save-path", type=Path, default=None, help="Optional: save summary to this file.")
    args = ap.parse_args()

    # Load model
    N, df_map = load_model(args.idf_model)

    # Read and tokenize corpus
    text = args.corpus_path.read_text(encoding="utf-8", errors="ignore")
    tokens = list(tokenize(text, keep_stopwords=args.keep_stopwords, min_len=args.min_len))

    if not tokens:
        summary = "No tokens after filtering.\n"
        print(summary, end="")
        if args.save_path:
            args.save_path.parent.mkdir(parents=True, exist_ok=True)
            args.save_path.write_text(summary, encoding="utf-8")
        return

    # Token-weighted average IDF (counts frequency)
    token_idfs = [idf(N, df_map.get(tok, 0)) for tok in tokens]
    avg_idf_token = statistics.fmean(token_idfs)

    # Type-weighted average IDF (unique tokens only)
    vocab = set(tokens)
    type_idfs = [idf(N, df_map.get(tok, 0)) for tok in vocab]
    avg_idf_type = statistics.fmean(type_idfs)

    # Prepare output
    summary = (
        f"Corpus: {args.corpus_path}\n"
        f"Model:  {args.idf_model}\n"
        f"Tokens kept: {len(tokens)}\n"
        f"Vocab size:  {len(vocab)}\n"
        f"Avg External IDF (token-weighted): {avg_idf_token:.4f}\n"
        f"Avg External IDF (type-weighted):  {avg_idf_type:.4f}\n"
    )

    print(summary, end="")

    # Save if requested
    if args.save_path:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        args.save_path.write_text(summary, encoding="utf-8")

if __name__ == "__main__":
    main()
