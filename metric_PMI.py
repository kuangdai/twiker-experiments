# metric_PMI.py
# Compute corpus-level mean PMI/NPMI over all observed co-occurrences.
# Usage:
#   python metric_PMI.py data/datasets/potter/data_files/train_new.txt --save-path corpus_metric/pmi/potter_mean.txt
#   python metric_PMI.py data/datasets/potter/data_files/train_new.txt --npmi --save-path corpus_metric/pmi/potter_mean_npmi.txt

import argparse
import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from tqdm import tqdm

TOKEN_RE = re.compile(r"[A-Za-z]+")

STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","that","this","these","those","is","am","are","was","were","be","been",
    "being","to","of","in","on","for","with","as","by","at","from","it","its","not","no","do","does","did","so","such","can",
    "could","should","would","will","shall","may","might","must","have","has","had","i","you","he","she","we","they","them",
    "him","her","my","your","our","their","me","us","which","who","whom","what","when","where","why","how"
}

def tokenize_stream(path: Path, keep_stopwords=False, min_len=2, chunk_chars=2_000_000):
    """Yield tokens from a large text file in character chunks to limit RAM."""
    total = path.stat().st_size
    read_so_far = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Reading") as pbar:
        while True:
            buf = f.read(chunk_chars)
            if not buf:
                break
            read_so_far += len(buf)
            pbar.update(len(buf))
            for m in TOKEN_RE.finditer(buf):
                t = m.group(0).lower()
                if len(t) < min_len:
                    continue
                if not keep_stopwords and t in STOPWORDS:
                    continue
                yield t

def build_counts(corpus_path: Path, window: int, keep_stopwords: bool, min_len: int, max_tokens: int | None):
    """
    Build unigram and symmetric window co-occurrence counts.
    Returns (unigrams, co_counts, total_pairs).
    """
    unigrams = Counter()
    co_counts = defaultdict(Counter)
    total_pairs = 0

    buf = deque(maxlen=window * 2 + 1)
    token_iter = tokenize_stream(corpus_path, keep_stopwords=keep_stopwords, min_len=min_len)

    processed = 0
    for tok in token_iter:
        unigrams[tok] += 1
        buf.append(tok)
        processed += 1
        if max_tokens and processed >= max_tokens:
            break
        if len(buf) == buf.maxlen:
            cidx = window
            center = buf[cidx]
            # add co-occurrences (unordered window, directed by center)
            for i, ctx in enumerate(buf):
                if i == cidx:
                    continue
                co_counts[center][ctx] += 1
                total_pairs += 1

    # Optionally drain tail (kept simple: skip; effect negligible on big corpora)
    return unigrams, co_counts, total_pairs

def mean_pmi(unigrams: Counter, co_counts: dict, total_pairs: int, use_npmi: bool):
    """
    Exact weighted mean PMI (or NPMI) over all observed (w,c) pairs.
    Weight is n_wc (the co-occurrence count).
    """
    if total_pairs == 0:
        return float("nan")

    weighted_sum = 0.0
    weight_total = 0

    for w, cnts in co_counts.items():
        n_w = unigrams.get(w, 0)
        if n_w == 0:
            continue
        for c, n_wc in cnts.items():
            n_c = unigrams.get(c, 0)
            if n_c == 0 or n_wc == 0:
                continue
            # PMI using counts: log( (n_wc * total_pairs) / (n_w * n_c) )
            num = n_wc * total_pairs
            den = n_w * n_c
            pmi_val = math.log(num / den)
            if use_npmi:
                pwc = n_wc / total_pairs
                denom = -math.log(pwc)
                # denom > 0 because pwc in (0,1]; safe since n_wc>0
                val = pmi_val / denom if denom > 0 else 0.0
            else:
                val = pmi_val

            weighted_sum += val * n_wc
            weight_total += n_wc

    return weighted_sum / weight_total if weight_total > 0 else float("nan")

def main():
    ap = argparse.ArgumentParser(description="Compute corpus-level mean PMI/NPMI over all co-occurrence pairs.")
    ap.add_argument("corpus_path", type=Path, help="Plain-text corpus file.")
    ap.add_argument("--window", type=int, default=5, help="Symmetric context window size (default: 5).")
    ap.add_argument("--min-len", type=int, default=2, help="Minimum token length (default: 2).")
    ap.add_argument("--keep-stopwords", action="store_true", help="Keep stopwords (default: drop).")
    ap.add_argument("--max-tokens", type=int, default=None, help="Optional cap on tokens processed (for speed/debug).")
    ap.add_argument("--npmi", action="store_true", help="Compute NPMI instead of PMI.")
    ap.add_argument("--min-count", type=int, default=5, help="Prune tokens with freq < min-count (default: 5).")
    ap.add_argument("--save-path", type=Path, default=None, help="Optionally save the printed summary here.")
    args = ap.parse_args()

    # Build raw counts
    unigrams, co_counts, total_pairs = build_counts(
        args.corpus_path, window=args.window, keep_stopwords=args.keep_stopwords,
        min_len=args.min_len, max_tokens=args.max_tokens
    )

    # Prune rare tokens to reduce PMI inflation and memory
    if args.min_count > 1:
        # tokens to keep
        keep = {w for w, c in unigrams.items() if c >= args.min_count}
        # filter unigrams
        unigrams = Counter({w: c for w, c in unigrams.items() if w in keep})
        # filter co-occurrences
        pruned = defaultdict(Counter)
        for w, cnts in co_counts.items():
            if w not in keep:
                continue
            for c, n in cnts.items():
                if c in keep:
                    pruned[w][c] = n
        co_counts = pruned

    mean_val = mean_pmi(unigrams, co_counts, total_pairs, use_npmi=args.npmi)
    metric = "NPMI" if args.npmi else "PMI"

    summary = (
        f"Corpus: {args.corpus_path}\n"
        f"Window: ±{args.window}\n"
        f"Keep stopwords: {bool(args.keep_stopwords)}\n"
        f"Min token length: {args.min_len}\n"
        f"Max tokens: {args.max_tokens if args.max_tokens else 'None'}\n"
        f"Min count (pruning): {args.min_count}\n"
        f"Vocabulary size (after pruning): {len(unigrams)}\n"
        f"Total co-occurrence pairs (raw): {total_pairs}\n"
        f"Mean {metric}: {mean_val:.6f}\n"
    )

    print(summary, end="")
    if args.save_path:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        args.save_path.write_text(summary, encoding="utf-8")

if __name__ == "__main__":
    main()
