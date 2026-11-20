import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json

from preprocess import clean_text, tokenize


# Custom PyTorch Dataset
class SentimentDataset(Dataset):

    def __init__(self, texts, labels, vocab, max_length=200):

        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        # Get a review
        text = self.texts[index]

        # Clean text
        text = clean_text(text)

        # Tokenize text
        tokens = tokenize(text)

        # Convert words into IDs
        encoded = [
            self.vocab.get(word, self.vocab["<UNK>"])
            for word in tokens
        ]

        # Truncate if longer than max_length
        encoded = encoded[:self.max_length]

        # Pad if shorter than max_length
        padding_length = self.max_length - len(encoded)

        encoded = encoded + [self.vocab["<PAD>"]] * padding_length

        # Convert into PyTorch tensor
        input_tensor = torch.tensor(encoded, dtype=torch.long)

        # Convert label into tensor
        label_tensor = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return input_tensor, label_tensor


# Execute the code
if __name__ == "__main__":

    # Load training data
    train_df = pd.read_csv("data/train.csv")

    # Load vocabulary
    with open("models/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Create Dataset
    dataset = SentimentDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        vocab=vocab,
        max_length=200
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # Get one batch
    batch_x, batch_y = next(iter(dataloader))

    print("Input batch shape:", batch_x.shape)
    print("Label batch shape:", batch_y.shape)

    print("\nFirst review tensor:")
    print(batch_x[0])

    print("\nFirst review label:")
    print(batch_y[0])