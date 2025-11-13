import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

df = pd.read_csv("./archive/Iris.csv")

print(df.head())

df = df.drop(columns=['Id'], errors='ignore')

X = df.drop(columns=['Species'])  
y = df['Species']                

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)  


model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")


cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nSample Predictions:")
print(pd.DataFrame({"Actual": y_test.values[:5], "Predicted": y_pred[:5]}))

joblib.dump(model, "iris_model.pkl")
print("\nModel saved as 'iris_model.pkl'.")


loaded_model = joblib.load("iris_model.pkl")
sample_pred = loaded_model.predict(X_test[:5])
print("\nPredictions from loaded model:", sample_pred)
