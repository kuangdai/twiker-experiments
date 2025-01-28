import argparse
import string

import nltk
import numpy as np
import spacy
import torch
import tqdm
from nltk.corpus import words

from transformers import GPT2Tokenizer

# Download NLTK words corpus
nltk.download('words')

# Load common words and SpaCy model
common_words = set(word.lower() for word in words.words())
nlp = spacy.load("en_core_web_sm")


def is_punctuation(char):
    """Check if a character is a punctuation mark."""
    return char in string.punctuation or ord(char) > 127  # Includes non-ASCII chars


def common_word(word):
    """Check if a word is in the common word list."""
    word = word.lower().strip(string.punctuation)
    return word in common_words


def get_pos(word):
    """Get the part-of-speech (POS) tag of a word using SpaCy."""
    doc = nlp(word)
    if doc and len(doc) > 0:
        return doc[0].pos_
    return "other"


def classify(token):
    """
    Classify a GPT-2 token into categories like punctuation, letter, affix, etc.
    """
    # Handle single-character tokens
    if len(token) == 1:
        if is_punctuation(token):
            return "punctuation"
        elif token.isalpha():
            return "letter"
        elif token.isdigit():
            return "num"
        return "other"

    # Handle tokens with "Ġ" prefix (full words or subwords)
    if token[0] == "Ġ":
        token = token[1:]  # Remove "Ġ" prefix
        if len(token) == 1:
            if is_punctuation(token):
                return "punctuation"
            elif token.isalpha() and token not in ["I", "a", "A"]:
                return "letter"
            elif token.isdigit():
                return "num"
        if not common_word(token):
            return "affix"
        return get_pos(token).lower()

    # Handle tokens without "Ġ" prefix (likely affixes)
    return "affix"


def load_dataset(name):
    """Load and tokenize the dataset."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dataset_path = f"./data/datasets/{name}/data_files/train_new.txt"
    with open(dataset_path, 'r', encoding="utf-8") as fs:
        text = fs.read()
    tokens = np.array(tokenizer.tokenize(text))
    token_ids = np.array(tokenizer.convert_tokens_to_ids(tokens))
    return text, tokens, token_ids


def process_weights(name, kernel_size, heads, temperature, using_keys, book_ids_unique):
    """Process the weight tensor and compute distances."""
    weights_path = f"./result_weights/{name}.pt"
    w = torch.load(weights_path)
    assert w.shape[1] == 2 * kernel_size * heads, "Mismatch in weight dimensions."

    w = w.reshape(-1, 2, heads, kernel_size)
    w = w[:, 0 if using_keys else 1, :, :]
    w = torch.mean(w, dim=1)  # Mean over heads
    w = torch.softmax(w / temperature, dim=-1)
    w = w[book_ids_unique]

    # Compute distance to the origin vector
    origin = torch.zeros(kernel_size)
    origin[kernel_size // 2] = 1.  # Center value is 1
    dist_all = (w - origin[None, :]).norm(dim=1).mean()
    return w, dist_all


def main():
    parser = argparse.ArgumentParser(description="Process GPT-2 tokens.")
    parser.add_argument("-n", "--name", type=str, required=True, help="Name of dataset.")
    parser.add_argument("-o", "--occurrence", type=int, default=5, help="Minimum token occurrence.")
    parser.add_argument("-t", "--temperature", type=float, default=0.4, help="Temperature in softmax.")
    parser.add_argument("-m", "--heads", type=int, default=1, help="Number of heads.")
    parser.add_argument("-s", "--kernel_size", type=int, default=3, help="Kernel size.")
    parser.add_argument("--using-keys", action="store_true", help="Trained with keys not values.")
    args = parser.parse_args()

    # Load dataset
    book_text, book_tokens, book_ids = load_dataset(args.name)
    print("Total tokens in dataset:", len(book_ids))

    # Extract unique tokens and their counts
    book_ids_unique, index_unique, counts_unique = np.unique(book_ids, return_index=True, return_counts=True)
    print("Unique tokens in dataset:", len(book_ids_unique))

    # Filter tokens based on occurrence
    filter_count = counts_unique > args.occurrence
    book_ids_unique = book_ids_unique[filter_count]
    index_unique = index_unique[filter_count]
    book_tokens_unique = book_tokens[index_unique]
    print("Frequent unique tokens in dataset:", len(book_ids_unique))

    # Process weights
    w, dist_all = process_weights(args.name, args.kernel_size, args.heads, args.temperature, args.using_keys,
                                  book_ids_unique)
    print("Mean distance to origin vector:", dist_all)

    # Identify non-ASCII characters
    non_ascii_characters = np.unique([char for char in book_text if ord(char) > 127])
    print("Non-ASCII characters:", non_ascii_characters)

    # Classify tokens
    classes = [classify(tok) for tok in tqdm.tqdm(book_tokens_unique, desc="Classifying tokens")]
    classes = np.array(classes)

    # Summarize token classes
    classes_unique = np.unique(classes)
    for cls in classes_unique:
        tokens_in_class = book_tokens_unique[classes == cls][:5]
        print(f"{cls}, {len(np.where(classes == cls)[0])}: {' '.join(tokens_in_class)}")

    # Analyze distances for specific classes
    origin = torch.zeros(args.kernel_size)
    origin[args.kernel_size // 2] = 1.  # Center value is 1
    super_classes = ["intj", "noun", "verb", "adj", "adv", "sconj", "aux", "adp", "pron", "cconj"]
    super_dist = []
    for cls in super_classes:
        idx = np.where(classes == cls)[0]
        if len(idx) > 0:
            dist = (w[idx] - origin[None, :]).norm(dim=1).mean()
        else:
            dist = float('nan')  # Handle empty classes
        super_dist.append(dist)
    super_dist = np.array(super_dist)
    np.savetxt(f"./result_weights/{args.name}.dist", super_dist, header=str(dist_all.item()))


if __name__ == "__main__":
    main()
