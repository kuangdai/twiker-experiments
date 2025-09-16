# build_idf_from_hf.py
import argparse, json, math, gzip, re
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

TOKEN_RE = re.compile(r"[A-Za-z]+")
STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","that","this","these","those","is","am","are","was","were","be","been",
    "being","to","of","in","on","for","with","as","by","at","from","it","its","not","no","do","does","did","so","such","can",
    "could","should","would","will","shall","may","might","must","have","has","had","i","you","he","she","we","they","them",
    "him","her","my","your","our","their","me","us","which","who","whom","what","when","where","why","how"
}

def tokenize(text, keep_stopwords=False, min_len=2):
    for m in TOKEN_RE.finditer(text):
        t = m.group(0).lower()
        if len(t) < min_len:
            continue
        if not keep_stopwords and t in STOPWORDS:
            continue
        yield t

def chunk_text(text, doc_chars):
    if doc_chars <= 0:
        yield text
    else:
        for i in range(0, len(text), doc_chars):
            yield text[i:i+doc_chars]

def main():
    ap = argparse.ArgumentParser(description="Build DF/IDF model from HuggingFace Wikipedia (streaming).")
    ap.add_argument("--snapshot", default="20220301.en",
                    help="Wikipedia snapshot config (e.g., 20220301.en).")
    ap.add_argument("--split", default="train", help="Dataset split (default: train).")
    ap.add_argument("--doc-chars", type=int, default=20000,
                    help="Chunk articles into pseudo-docs of this many chars (default 20000).")
    ap.add_argument("--keep-stopwords", action="store_true", help="Keep stopwords in DF.")
    ap.add_argument("--min-len", type=int, default=2, help="Min token length (default 2).")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Optional limit on number of pseudo-docs for quick builds/dev.")
    ap.add_argument("--out", type=Path, required=True, help="Output path (.json or .json.gz).")
    args = ap.parse_args()

    # Stream dataset (no full download to disk required)
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    df = Counter()
    N = 0  # number of pseudo-docs
    pbar = tqdm(desc="Building DF", unit="doc")
    for ex in ds:
        text = ex.get("text") or ""
        # split article into pseudo-docs for more stable DF statistics
        for chunk in chunk_text(text, args.doc_chars):
            vocab = set(tokenize(chunk, keep_stopwords=args.keep_stopwords, min_len=args.min_len))
            if not vocab:
                continue
            df.update(vocab)
            N += 1
            pbar.update(1)
            if args.max_docs is not None and N >= args.max_docs:
                break
        if args.max_docs is not None and N >= args.max_docs:
            break
    pbar.close()

    model = {"N": N, "df": dict((k, int(v)) for k, v in df.items())}

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.out.suffix == ".gz":
        with gzip.open(args.out, "wt", encoding="utf-8") as g:
            json.dump(model, g)
    else:
        args.out.write_text(json.dumps(model), encoding="utf-8")

    print(f"\nSaved model to {args.out}  (N={N} pseudo-docs, |V|={len(df)})")

if __name__ == "__main__":
    main()
