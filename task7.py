import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Stopwords download kar raha hai (pehli dafa mein required hota hai)
nltk.download("stopwords")

# ---------------------------
# 1. Training Dataset
# Ye hamara training data hai.
# Har item mein ek sentence hai aur uska sentiment label (positive/negative)
# ---------------------------
data = [
    ("I love this product", "positive"),
    ("This is amazing", "positive"),
    ("I am very happy", "positive"),
    ("Absolutely fantastic service", "positive"),
    ("Worst experience ever", "negative"),
    ("This is bad", "negative"),
    ("I hate this", "negative"),
    ("Very disappointing", "negative"),
    ("Terrible product", "negative"),
    ("Not good at all", "negative"),
]

#  Yahan hum data ko do lists me split kar rahe hain:
# texts → sirf sentences
# labels → sentiments (positive/negative)
texts = [t[0] for t in data]
labels = [t[1] for t in data]

# ---------------------------
# 2. Stopwords removal + TF-IDF vectorization

# Stopwords = common lafz jo model ko confuse karte hain (is, the, a, etc.)
# TF-IDF = text ko numbers me convert karta hai taki ML model samajh sake
# ---------------------------
stop_words = stopwords.words("english")

vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(texts)  # Roman Urdu: Ab sentences numeric vectors ban gaye

# ---------------------------
# 3. Train-Test Split

# Training data = model ko sikhane ke liye
# Testing data = model ko check karne ke liye
# test_size=0.3 → 30% testing, 70% training
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

# ---------------------------
# 4. Model Training

# Logistic Regression ek ML algorithm hai jo sentiment predict karta hai
# model.fit() = yahan model actual training karta hai
# ---------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------
# 5. Model Evaluation

# y_pred = model test data ka sentiment predict karega
# accuracy_score = kitna sahi predict kya, percentage form me
# ---------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# ---------------------------
# 6. Real-time Prediction Loop

# User se sentence lega
# Pehle usay vectorizer se numbers me convert karega
# Model uska sentiment predict karega
# ---------------------------
print("\nEnter sentences to predict sentiment (type 'quit' to exit):")
while True:
    text = input(">>> ")
    if text.lower() == "quit":
        break

    # input text ko numeric vector bana rahe hain 
    text_processed = vectorizer.transform([text])

    #  model se prediction nikal rahe hain
    pred = model.predict(text_processed)[0]

    print("Sentiment:", pred.capitalize())
