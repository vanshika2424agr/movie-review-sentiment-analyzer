# 🎬 Movie Review Sentiment Analyzer

A deep learning-based sentiment analysis application built using **PyTorch** and **Streamlit** that predicts whether a movie review expresses a Positive or Negative sentiment.

## 🚀 Features

- Text preprocessing and tokenization
- Custom vocabulary construction
- Word embeddings using PyTorch
- Neural network-based sentiment classification
- Model training and validation
- Evaluation using Accuracy, Precision, Recall and F1-score
- Confusion matrix visualization
- Interactive Streamlit web application
- Prediction confidence display

## 🧠 Model Architecture

The model uses the following architecture:

1. Embedding Layer — converts word indices into dense vectors.
2. Mean Pooling — combines word embeddings into a review representation while ignoring padding.
3. Fully Connected Layer — 64 to 128 dimensions.
4. ReLU Activation — introduces non-linearity.
5. Output Layer — predicts Negative or Positive sentiment.

## 📊 Dataset

- Dataset: IMDB Movie Reviews
- Training dataset: 25,000 reviews (80% used for training and 20% for validation)
- Test dataset: 25,000 reviews
- Classes: Negative (0), Positive (1)

## 📈 Test Results

| Metric | Score |
|---|---:|
| Accuracy | 84.28% |
| Precision | 85.29% |
| Recall | 82.84% |
| F1 Score | 84.05% |

## 🛠️ Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## 📂 Project Structure

```text
sentiment_classifier/
├── data/
├── models/
│   ├── vocab.json
│   ├── sentiment_model.pth
│   └── confusion_matrix.png
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sentiment_classifier.git
cd sentiment_classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

## 🎯 Sample Prediction

Input:

> This movie was absolutely fantastic. The acting was brilliant and the story was engaging.

Output:

**Positive Sentiment 😊**

## 👩‍💻 Project Purpose

This project was built to learn and implement an end-to-end NLP and deep learning pipeline using PyTorch, from preprocessing and model training to evaluation and deployment-ready application development.
