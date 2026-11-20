import re
import pandas as pd
from collections import Counter
import json


# 1. Clean text
def clean_text(text):
    text = text.lower()
    text = re.findall(r"\b[\w']+\b", text)
    text = " ".join(text)
    return text


# 2. Tokenize text
def tokenize(text):
    return text.split()


# 3. Build vocabulary
def build_vocab(texts, max_vocab_size=20000):

    counter = Counter()

    for text in texts:
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        counter.update(tokens)

    # Reserve special tokens
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    # Select most frequent words
    most_common_words = counter.most_common(max_vocab_size - 2)

    for word, frequency in most_common_words:
        vocab[word] = len(vocab)

    return vocab


# 4. Encode text into numbers
def encode_text(text, vocab):

    cleaned = clean_text(text)
    tokens = tokenize(cleaned)

    encoded = [
        vocab.get(word, vocab["<UNK>"])
        for word in tokens
    ]

    return encoded


# 5. Execute the code
if __name__ == "__main__":

    # Load training dataset
    train_df = pd.read_csv("data/train.csv")

    # Build vocabulary using training reviews only
    vocab = build_vocab(train_df["text"], max_vocab_size=20000)

    print("Vocabulary size:", len(vocab))

    # Test on sample review
    sample = "This movie was absolutely fantastic!"

    print("\nOriginal sentence:")
    print(sample)

    print("\nCleaned sentence:")
    print(clean_text(sample))

    print("\nTokens:")
    print(tokenize(clean_text(sample)))

    print("\nEncoded sentence:")
    print(encode_text(sample, vocab))

    # Save vocabulary
    with open("models/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print("\nVocabulary saved successfully!")