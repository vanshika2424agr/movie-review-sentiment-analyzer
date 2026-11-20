from datasets import load_dataset
import pandas as pd

# Download IMDB dataset
dataset = load_dataset("stanfordnlp/imdb")

# Convert training data to DataFrame
train_df = pd.DataFrame(dataset["train"])

# Convert testing data to DataFrame
test_df = pd.DataFrame(dataset["test"])

# Save training data
train_df.to_csv("data/train.csv", index=False)

# Save testing data
test_df.to_csv("data/test.csv", index=False)

print("Dataset downloaded successfully!")

print("\nTraining data:")
print(train_df.head())

print("\nTesting data:")
print(test_df.head())

print("\nTraining shape:", train_df.shape)
print("Testing shape:", test_df.shape)