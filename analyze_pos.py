import argparse
import json
import re

import spacy
import tqdm
from nltk.tokenize import sent_tokenize

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
nlp = spacy.load("en_core_web_sm")


def is_only_ascii_letters(s):
    return bool(re.fullmatch(r"[a-zA-Z]+", s))


def process_sentence(sentence):
    """Ensure only one space between words and no space before punctuation/symbols."""
    sentence = re.sub(r'\s+', ' ', sentence)  # Replace multiple spaces with a single space
    sentence = re.sub(r'\s([?.!,;:])', r'\1', sentence)  # Remove space before punctuation
    return sentence.strip()


def analyze_sentence(sentence):
    sentence = process_sentence(sentence)

    doc = nlp(sentence)

    # Get words and their POS
    words = []
    pos_tags = []
    for word in doc:
        words.append(word.text)
        pos_tags.append(word.pos_)

    # Encode the sentence using the tokenizer
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Map words to tokens and assign POS
    tokens_good = []
    token_ids_good = []
    token_pos = []
    word_index = 0

    for token, token_id in zip(tokens, token_ids):
        token = token.replace("Ġ", "")  # Remove GPT-2's word boundary marker
        if not is_only_ascii_letters(token):
            continue  # Too complicated if considering symbols

        # Find next word
        word_index_use = word_index
        found = False
        while word_index_use < len(words):
            if token == words[word_index_use]:
                found = True
                break
            word_index_use += 1
        if not found:
            continue
        word_index = word_index_use

        # Save match
        token_pos.append(pos_tags[word_index].lower())
        tokens_good.append(token)
        token_ids_good.append(token_id)
        word_index += 1
    return tokens_good, token_ids_good, token_pos


def load_to_sentence(name):
    """Load and tokenize the dataset."""
    dataset_path = f"./data/datasets/{name}/data_files/train_new.txt"
    with open(dataset_path, 'r', encoding="utf-8") as fs:
        text = fs.read()
    sentences = sent_tokenize(text)
    return sentences


def main():
    parser = argparse.ArgumentParser(description="Analysis Token POS.")
    parser.add_argument("-n", "--name", type=str, required=True, help="Name of dataset.")
    args = parser.parse_args()

    # Load to sentence
    sentences = load_to_sentence(args.name)

    # Analyze sentence
    pos_dict = {}
    for sent in tqdm.tqdm(sentences):
        tokens, token_ids, token_pos = analyze_sentence(sent)
        for token, token_id, pos in zip(tokens, token_ids, token_pos):
            if token_id not in pos_dict:
                pos_dict[token_id] = {}
                pos_dict[token_id]["token"] = token
                pos_dict[token_id]["poses"] = []
            pos_dict[token_id]["poses"].append(pos)

    # Save
    with open(f"pos/{args.name}.json", "w") as file:
        json.dump(pos_dict, file)


if __name__ == "__main__":
    main()
