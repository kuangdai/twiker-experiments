import argparse
import string

import nltk
import numpy as np
import spacy
import torch
import tqdm
from nltk.corpus import words

from transformers import GPT2Tokenizer

nltk.download('words')
common_words = set(words.words())
nlp = spacy.load("en_core_web_sm")


def common_word(word):
    return word.lower() in common_words


def get_pos(word):
    doc = nlp(word)
    for token in doc:
        return token.pos_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process.")
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="Name of dataset.")
    parser.add_argument("-t", "--temperature", type=float, default=0.4,
                        help="Temperature in softmax.")
    parser.add_argument("-o", "--occurrence", type=int, default=5,
                        help="Minimum occurrence.")
    args = parser.parse_args()

    # Tokenization
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    with open(f"./data/datasets/{args.name}/data_files/train_new.txt", 'r') as fs:
        book_text = fs.read()
    book_tokens = np.array(tokenizer.tokenize(book_text))
    book_ids = np.array(tokenizer.convert_tokens_to_ids(book_tokens))
    print("Total tokens in dataset:", len(book_ids))

    # unique
    book_ids_unique, index_unique, counts_unique = np.unique(book_ids, return_index=True, return_counts=True)
    print("Unique tokens in dataset", len(book_ids_unique))

    # Filter by count
    filter_count = counts_unique > args.occurrence
    book_ids_unique = book_ids_unique[filter_count]
    index_unique = index_unique[filter_count]
    counts_unique = counts_unique[filter_count]
    book_tokens_unique = book_tokens[index_unique]
    print("Frequent Unique tokens in dataset", len(book_ids_unique))

    # Weights
    w = torch.load(f"./weights/{args.name}.pt")
    w = w[:, 3:]  # value
    w = torch.softmax(w / args.temperature, dim=-1)
    w = w[book_ids_unique]
    dist_all = (w - torch.tensor([0., 1., 0.])[None, :]).norm(dim=1).mean()
    print("Mean distance to 010:", dist_all)

    # NLP processing
    non_ascii_characters = np.unique([char for char in book_text if ord(char) > 127])
    print("Non-ASCII characters:", non_ascii_characters)


    def is_punctuation(char):
        return char in string.punctuation or char in non_ascii_characters


    classes = []
    for tok in tqdm.tqdm(book_tokens_unique):
        if len(tok) == 1:
            if is_punctuation(tok):
                classes.append("punctuation")
            elif tok.isalpha():
                classes.append("letter")
            elif tok.isdigit():
                classes.append("num")
            else:
                classes.append("other")
            continue

        if tok[0] != "Ġ":
            classes.append("affix")
            continue

        tok = tok[1:]
        if len(tok) == 1:
            if is_punctuation(tok):
                classes.append("punctuation")
                continue
            if tok.isalpha() and tok not in ["I", "a", "A"]:  # here we check single-char words
                classes.append("letter")
                continue
            elif tok.isdigit():
                classes.append("num")
                continue
        if not common_word(tok):
            classes.append("affix")
            continue
        classes.append(get_pos(tok).lower())
    classes = np.array(classes)
    classes_unique = np.unique(classes)
    for cls in classes_unique:
        print(f"{cls}, {len(np.where(classes == cls)[0])}: "
              f"{' '.join(book_tokens_unique[classes == cls][:5])}")

    # classes for analysis
    super_classes = ["noun", "verb", "adj", "adv", "sconj", "aux", "pron", "adp", "cconj"]
    super_dist = []
    for cls in super_classes:
        idx = np.where(classes == cls)[0]
        dist = (w[idx] - torch.tensor([0., 1., 0.])[None, :]).norm(dim=1).mean()
        super_dist.append(dist)
    super_dist = np.array(super_dist)
    np.savetxt(f"./results/{args.name}", super_dist, header=str(dist_all.item()))
