# pip install -U spacy tqdm
# python -m spacy download en_core_web_sm

import argparse
import statistics
from pathlib import Path
from collections import deque
from typing import Optional, List

import spacy
from tqdm import tqdm


def split_text(text: str, chunk_size: int) -> List[str]:
    """Split text into ~equal-sized character chunks (no overlap)."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def sentence_depth_bfs(sent, drop_punct: bool = True, min_tokens: int = 3) -> Optional[int]:
    """
    Compute dependency tree depth for a sentence using BFS from root(s).
    Depth counts nodes (root=1). Returns None if sentence too short.
    """
    tokens = [t for t in sent if not (drop_punct and t.is_punct)]
    if len(tokens) < min_tokens:
        return None

    roots = [t for t in tokens if t.head is t]
    if not roots:
        roots = [t for t in tokens if t.dep_ == "ROOT"]
    if not roots:
        return None

    max_depth = 0
    q = deque((r, 1) for r in roots if not (drop_punct and r.is_punct))

    while q:
        tok, depth = q.popleft()
        if drop_punct and tok.is_punct:
            continue
        if depth > max_depth:
            max_depth = depth
        for child in tok.children:
            if drop_punct and child.is_punct:
                continue
            q.append((child, depth + 1))

    return max_depth if max_depth > 0 else None


def compute_mean_parse_depth(
    corpus_path: str,
    chunk_size: int = 20000,
    batch_size: int = 4,
    min_tokens: int = 3,
    drop_punct: bool = True,
) -> str:
    """Compute mean dependency parse depth across a large corpus with chunking & batching.
    Returns the exact summary string that is printed.
    """
    nlp = spacy.load("en_core_web_sm")
    if "senter" not in nlp.pipe_names:
        nlp.enable_pipe("senter")

    text = Path(corpus_path).read_text(encoding="utf-8")
    chunks = split_text(text, chunk_size)

    depths = []
    total_sentences = 0

    for doc in tqdm(nlp.pipe(chunks, batch_size=batch_size), total=len(chunks), desc="Processing chunks"):
        for sent in doc.sents:
            total_sentences += 1
            d = sentence_depth_bfs(sent, drop_punct=drop_punct, min_tokens=min_tokens)
            if d is not None:
                depths.append(d)

    if not depths:
        summary = "No valid sentences found.\n"
        print(summary, end="")
        return summary

    mean_depth = statistics.fmean(depths)
    median_depth = statistics.median(depths)
    std_depth = statistics.pstdev(depths)

    summary = (
        f"\nCorpus: {corpus_path}\n"
        f"# Chunks: {len(chunks)}\n"
        f"# Sentences processed: {total_sentences}\n"
        f"# Sentences used: {len(depths)}\n"
        f"Mean dependency depth: {mean_depth:.3f}\n"
        f"Median depth: {median_depth:.3f}\n"
        f"Std dev: {std_depth:.3f}\n"
    )
    print(summary, end="")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Compute mean dependency parse depth for a corpus.")
    ap.add_argument("corpus_path", help="Path to plain-text corpus file (e.g., corpus.txt)")
    ap.add_argument("--chunk-size", type=int, default=20000, help="Characters per chunk (default: 20000)")
    ap.add_argument("--batch-size", type=int, default=4, help="Chunks processed per batch (default: 4)")
    ap.add_argument("--min-tokens", type=int, default=3, help="Min non-punct tokens per sentence (default: 3)")
    ap.add_argument(
        "--keep-punct",
        action="store_true",
        help="Include punctuation tokens in depth calculation (default: drop punctuation)",
    )
    ap.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="If provided, save exactly the printed summary to this file path.",
    )
    args = ap.parse_args()

    summary = compute_mean_parse_depth(
        corpus_path=args.corpus_path,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        min_tokens=args.min_tokens,
        drop_punct=not args.keep_punct,
    )

    if args.save_path:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        args.save_path.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
