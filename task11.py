

import sys
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))

def simple_preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove urls
    text = re.sub(r"[^a-z\s]", " ", text)  # remove non-letters
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def load_dataset(path):
    df = pd.read_csv(path)
    # Heuristics for text column
    text_col = None
    for col in ["text", "title", "article", "content"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        # take the first non-label text-like column
        for c in df.columns:
            if c.lower() != "label":
                text_col = c
                break
    assert text_col is not None, "No text column found in dataset."
    assert "label" in df.columns, "Dataset must contain a 'label' column."

    df = df[[text_col, "label"]].rename(columns={text_col: "text"})
    df["text"] = df["text"].astype(str).apply(simple_preprocess)
    # Normalize labels to 0/1
    if df["label"].dtype == object:
        df["label"] = df["label"].map(lambda x: 1 if str(x).lower() in ("fake","1","true","yes") else 0)
    else:
        df["label"] = df["label"].astype(int)
    return df

def train_and_eval(df):
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    preds = clf.predict(X_test_tfidf)
    print("Evaluation on test set:")
    print(classification_report(y_test, preds, digits=4))
    print("Accuracy:", accuracy_score(y_test, preds))

    return vectorizer, clf

def interactive_predict(vectorizer, clf):
    print("\nEnter a headline/text to classify (or 'quit' to exit):")
    while True:
        text = input(">> ").strip()
        if text.lower() in ("quit", "exit"):
            break
        proc = simple_preprocess(text)
        vec = vectorizer.transform([proc])
        pred = clf.predict(vec)[0]
        proba = clf.predict_proba(vec)[0].max() if hasattr(clf, "predict_proba") else None
        label_str = "FAKE" if pred == 1 else "REAL"
        if proba is not None:
            print(f"Prediction: {label_str} (confidence {proba:.2f})")
        else:
            print(f"Prediction: {label_str}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python task11_fake_news.py path/to/dataset.csv")
        sys.exit(1)
    path = sys.argv[1]
    df = load_dataset(path)
    vec, clf = train_and_eval(df)
    interactive_predict(vec, clf)

if __name__ == "__main__":
    main()
