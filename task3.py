import pandas as pd # frame manipulation and data analysis
from sklearn.model_selection import train_test_split # for splitting data into training and testing sets
from sklearn.linear_model import LogisticRegression # for logistic regression model
from sklearn.tree import DecisionTreeClassifier # for decision tree model in case we use it
from sklearn.metrics import accuracy_score, confusion_matrix # for evaluating model performance
import joblib # for saving and loading the trained model

df = pd.read_csv("./archive/Iris.csv")

print(df.head()) # it prints first 5 rows of dataset
print(df.info())  # it displays the datatype of each column and non-null counts and memory usage
print(df.describe()) # it displays the statistical summary of numerical columns    

df = df.drop(columns=['Id'], errors='ignore')

X = df.drop(columns=['Species'])  # it is target variable which we need to predict that's why we are removing it from features so that model predicts without cheating
y = df['Species']                

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)  #typically iris dataset is linearly separable so logistic regression works well and we use 200 max iterations to ensure convergence


model.fit(X_train, y_train) # here model is trained here X-train and y-train are training data features and labels respectively

y_pred = model.predict(X_test) # here model makes predictions on test data features


accuracy = accuracy_score(y_test, y_pred) # for calculating accuracy of model by comparing actual labels and predicted labels
print(f"Model Accuracy: {accuracy*100:.2f}%")


cm = confusion_matrix(y_test, y_pred) # for creating confusion matrix to evaluate accuracy of classification
print("\nConfusion Matrix:")
print(cm)

print("\nSample Predictions:")
print(pd.DataFrame({"Actual": y_test.values[:5], "Predicted": y_pred[:5]})) # displaying first 5 actual and predicted values for comparison

joblib.dump(model, "iris_model.pkl") # saving the trained model to a file
print("\nModel saved as 'iris_model.pkl'.")


loaded_model = joblib.load("iris_model.pkl") # loading the saved model from file
sample_pred = loaded_model.predict(X_test[:5]) # making predictions on first 5 test samples using loaded model
print("\nPredictions from loaded model:", sample_pred)
