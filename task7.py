
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------
#  Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords


data = {
    "review": [
        "I loved this movie, it was amazing!",
        "Terrible movie. I will not recommend.",
        "It was okay, not great but not bad either.",
        "What a fantastic experience, truly enjoyed it!",
        "Worst movie ever, completely waste of time."
    ],
    "sentiment": ["positive", "negative", "positive", "positive", "negative"]
}
df = pd.DataFrame(data)

# --------------------------

def clean_text(text):
    text = text.lower() 
    text = re.sub(r'[^a-z\s]', '', text)  
    text = ' '.join([w for w in text.split() if w not in stopwords.words('english')])
    return text

df['cleaned'] = df['review'].apply(clean_text)

# --------------------------
#  Convert text to numeric features
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['cleaned'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# --------------------------
#  Split data & train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# Evaluate model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# --------------------------
# 7️⃣ CLI to test new sentences
print("\nEnter sentences to predict sentiment (type 'quit' to exit):")
while True:
    text = input(">>> ")
    if text.lower() == 'quit':
        break
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    print("Sentiment:", "Positive" if prediction == 1 else "Negative")
