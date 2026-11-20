import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import json

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from dataset import SentimentDataset
from model import SentimentModel


# 1. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Load test data
test_df = pd.read_csv("data/test.csv")

print("Test samples:", len(test_df))


# 3. Load vocabulary
with open("models/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)


# 4. Create test dataset
test_dataset = SentimentDataset(
    texts=test_df["text"].tolist(),
    labels=test_df["label"].tolist(),
    vocab=vocab,
    max_length=200
)


# 5. Create DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# 6. Create model
model = SentimentModel(
    vocab_size=len(vocab),
    embedding_dim=64,
    hidden_dim=128
).to(device)


# 7. Load trained weights
model.load_state_dict(
    torch.load("models/sentiment_model.pth", map_location=device)
)

model.eval()


# 8. Make predictions
all_predictions = []
all_labels = []

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(batch_y.cpu().tolist())


# 9. Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


# 10. Print results
print("\n===== TEST RESULTS =====")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# 11. Classification report
print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Negative", "Positive"],
        zero_division=0
    )
)


# 12. Confusion matrix
print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(all_labels, all_predictions)

print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("IMDB Sentiment Classifier - Confusion Matrix")

plt.tight_layout()

plt.savefig("models/confusion_matrix.png", dpi=300)

plt.show()
