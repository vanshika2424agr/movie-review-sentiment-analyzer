
import torch
import torch.nn as nn


class SentimentModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()

        # 1. Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # 2. Hidden layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)

        # 3. Activation function
        self.relu = nn.ReLU()

        # 4. Output layer
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):

        # Convert token IDs into embeddings
        embeddings = self.embedding(x)

        # Create mask to ignore padding tokens
        mask = (x != 0).unsqueeze(-1)

        # Remove padding contribution
        embeddings = embeddings * mask

        # Calculate average of non-padding embeddings
        sum_embeddings = embeddings.sum(dim=1)

        valid_tokens = mask.sum(dim=1).clamp(min=1)

        pooled = sum_embeddings / valid_tokens

        # Hidden layer
        hidden = self.fc1(pooled)

        # ReLU
        hidden = self.relu(hidden)

        # Output logits
        logits = self.fc2(hidden)

        return logits

if __name__ == "__main__":

    vocab_size = 20000

    model = SentimentModel(vocab_size)

    dummy_input = torch.randint(
        0,
        vocab_size,
        (32, 200)
    )

    # Add some padding tokens for testing
    dummy_input[:, -10:] = 0

    output = model(dummy_input)

    print("Output shape:", output.shape)
    print("\nFirst prediction scores:")
    print(output[0])

    print("\nTotal model parameters:",
          sum(p.numel() for p in model.parameters()))