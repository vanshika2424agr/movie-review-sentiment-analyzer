import streamlit as st
import torch
import json

from src.model import SentimentModel
from src.preprocess import clean_text, tokenize


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load vocabulary
with open("models/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)


# Load model
model = SentimentModel(
    vocab_size=len(vocab),
    embedding_dim=64,
    hidden_dim=128
).to(device)

model.load_state_dict(
    torch.load("models/sentiment_model.pth", map_location=device)
)

model.eval()


# Prediction function
def predict_sentiment(review):

    text = clean_text(review)
    tokens = tokenize(text)

    encoded = [
        vocab.get(word, vocab["<UNK>"])
        for word in tokens
    ]

    max_length = 200

    encoded = encoded[:max_length]

    padding_length = max_length - len(encoded)

    encoded = encoded + [vocab["<PAD>"]] * padding_length

    input_tensor = torch.tensor(
        [encoded],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = outputs.argmax(dim=1).item()

    confidence = probabilities[0][prediction].item()

    if prediction == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, confidence


# UI
st.set_page_config(
    page_title="Sentiment Classifier",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analyzer")

st.write(
    "Enter a movie review below and our AI model "
    "will predict whether the sentiment is Positive or Negative."
)

review = st.text_area(
    "Enter your movie review:",
    height=180,
    placeholder="This movie was amazing..."
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:
        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction")

        if sentiment == "Positive":
            st.success(f"Positive Sentiment 😊")
        else:
            st.error(f"Negative Sentiment 😞")

        st.metric(
            label="Model Confidence",
            value=f"{confidence * 100:.2f}%"
        )