"""
Fake News Cop (fake_news_cop.py)
--------------------------------
A simple machine learning classifier built with scikit-learn and NLTK 
to detect fake news based on title and text content using TF-IDF and Logistic Regression.
"""

import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Download required NLTK stopwords quietly
nltk.download("stopwords", quiet=True)

# Constants
STOPWORDS = set(stopwords.words("english"))
DATA_PATH = os.path.join(os.path.dirname(__file__), "news_dataset.csv")


def clean_text(text: str) -> str:
    """
    Cleans raw text by converting to lowercase, removing URLs,
    keeping only alphabetic characters, and filtering out stopwords and short words.
    """
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # Remove stopwords and words with length <= 2
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)


def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the CSV dataset, handles missing values, and creates 
    a combined 'content' column with cleaned text.
    """
    df = pd.read_csv(path)
    # Combine title and text, fill NaN with empty strings, then clean
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)
    return df


def train_and_evaluate(df: pd.DataFrame):
    """
    Splits the dataset, vectorizes text using TF-IDF, trains a Logistic Regression model,
    and prints evaluation metrics (accuracy, classification report, confusion matrix).
    """
    # Stratified train-test split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Initialize TF-IDF Vectorizer with unigrams and bigrams
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate performance on test set
    predictions = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
    print(classification_report(y_test, predictions, target_names=["Fake (0)", "Real (1)"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    return model, vectorizer


def predict_news(model, vectorizer, title: str, text: str) -> str:
    """
    Predicts whether a given news article (title + text) is Real or Fake.
    """
    content = clean_text(f"{title} {text}")
    vec = vectorizer.transform([content])
    label = model.predict(vec)[0]
    return "Real" if label == 1 else "Fake"


if __name__ == "__main__":
    # Load dataset and train model
    dataset = load_dataset(DATA_PATH)
    model, vectorizer = train_and_evaluate(dataset)

    # Interactive CLI loop for testing custom news samples
    print("\nEnter news details to check (leave title empty to exit).")
    while True:
        sample_title = input("\nTitle: ")
        if not sample_title.strip():
            print("Exiting program.")
            break
        sample_text = input("Text: ")
        print(f"Prediction: {predict_news(model, vectorizer, sample_title, sample_text)}")