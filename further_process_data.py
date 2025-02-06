import re
from pathlib import Path

import numpy as np

from transformers import GPT2Tokenizer


def split_into_sentences(text):
    # Split by sentence-ending punctuation with a regex
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text.strip())
    return sentences


def create_datasets_with_strict_token_limit(sentences, tokenizer, max_tokens=1000, max_data=2000, seed=42):
    np.random.seed(seed)
    datasets = []
    lengths = []

    # Randomly select 2000 unique starting indices
    start_sentence_ids = np.random.choice(len(sentences), len(sentences), replace=False)

    for start_idx in start_sentence_ids:
        current_entry = []
        current_tokens = 0

        # Aggregate sentences starting from the sampled index
        for idx in range(start_idx, len(sentences)):
            sentence = sentences[idx]
            tokenized_sentence = tokenizer.encode(sentence)
            token_count = len(tokenized_sentence)
            current_entry.append(sentence)
            current_tokens += token_count
            if current_tokens >= max_tokens:
                break

        # data
        data_sentence = " ".join(current_entry)
        length = len(tokenizer.encode(data_sentence))
        if length < max_tokens // 2:
            continue
        datasets.append(data_sentence)
        lengths.append(len(tokenizer.encode(data_sentence)))
        if len(datasets) == max_data:
            break
    return datasets, np.array(lengths)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    folder_path = Path('data/datasets')
    ds_paths = [f for f in folder_path.iterdir() if not f.is_file()]

    dataset_dict = {}

    for ds_path in ds_paths:
        dataset_name = ds_path.name
        if not "stone_" in dataset_name:
            continue
        with open(ds_path / "data_files/full.txt", 'r') as fs:
            train_text = fs.read()

        # Preprocess text
        train_text = train_text.replace("\n", " ")

        # Split text into sentences using nltk's sent_tokenize
        train_sentences = split_into_sentences(train_text)

        # Create datasets by aggregating sentences
        dataset, lengths = create_datasets_with_strict_token_limit(
            train_sentences, tokenizer, max_tokens=1000, max_data=2000, seed=42
        )
        dataset_dict[dataset_name] = dataset

        # Print summary
        print(
            f"{dataset_name}: size={len(dataset)}, "
            f"max_token_length={lengths.max()}, min_token_length={lengths.min()}, ave_token_length={lengths.mean()}"
        )

        # Save the dataset to train_new.txt, each entry on a new line
        with open(ds_path / "data_files/train_new.txt", "w") as output_file:
            output_file.write("\n".join(dataset))

        # Create datasets by aggregating sentences
        dataset, lengths = create_datasets_with_strict_token_limit(
            train_sentences, tokenizer, max_tokens=1000, max_data=200, seed=43
        )
        dataset_dict[dataset_name] = dataset

        # Print summary
        print(
            f"{dataset_name}: size={len(dataset)}, "
            f"max_token_length={lengths.max()}, min_token_length={lengths.min()}, ave_token_length={lengths.mean()}"
        )

        # Save the dataset to train_new.txt, each entry on a new line
        with open(ds_path / "data_files/eval.txt", "w") as output_file:
            output_file.write("\n".join(dataset))
