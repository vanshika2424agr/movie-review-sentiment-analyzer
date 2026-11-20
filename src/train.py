
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
import json

from sklearn.model_selection import train_test_split

from dataset import SentimentDataset
from model import SentimentModel


# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 2. Load dataset
df = pd.read_csv("data/train.csv")

# Split into training and validation datasets
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))


# 3. Load vocabulary
with open("models/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)


# 4. Create Dataset objects
train_dataset = SentimentDataset(
    texts=train_df["text"].tolist(),
    labels=train_df["label"].tolist(),
    vocab=vocab,
    max_length=200
)

val_dataset = SentimentDataset(
    texts=val_df["text"].tolist(),
    labels=val_df["label"].tolist(),
    vocab=vocab,
    max_length=200
)


# 5. Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# 6. Create model
model = SentimentModel(
    vocab_size=len(vocab),
    embedding_dim=64,
    hidden_dim=128
).to(device)


# 7. Loss function
criterion = nn.CrossEntropyLoss()


# 8. Optimizer
optimizer = Adam(
    model.parameters(),
    lr=0.001
)


# 9. Training configuration
epochs = 5


# 10. Training loop
for epoch in range(epochs):

    model.train()

    total_train_loss = 0
    correct_train = 0
    total_train = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_x)

        # Calculate loss
        loss = criterion(outputs, batch_y)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_train_loss += loss.item()

        # Calculate training accuracy
        predictions = outputs.argmax(dim=1)

        correct_train += (predictions == batch_y).sum().item()
        total_train += batch_y.size(0)

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train


    # 11. Validation
    model.eval()

    total_val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            total_val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct_val += (predictions == batch_y).sum().item()
            total_val += batch_y.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val


    # 12. Print results
    print(f"\nEpoch {epoch + 1}/{epochs}")

    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")


# 13. Save model
torch.save(model.state_dict(), "models/sentiment_model.pth")

print("\nModel saved successfully!")
