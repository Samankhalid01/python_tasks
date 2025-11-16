

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def load_and_prepare():
    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns=["MedHouseVal"])
    y = data.frame["MedHouseVal"]

    # Simple missing value handling (dataset has no missing values typically)
    X = X.fillna(X.median())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Choose model: RandomForest (uncomment LinearRegression if you want baseline)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return model, preds, mae, r2

def save_predictions(y_test, preds):
    df = pd.DataFrame({"actual": y_test.values, "predicted": preds})
    df.to_csv("housing_predictions.csv", index=False)
    print("Saved predictions to housing_predictions.csv")

def plot_preds(y_test, preds):
    plt.figure(figsize=(7,7))
    plt.scatter(y_test, preds, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linewidth=2)
    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title("Predicted vs Actual — California Housing")
    plt.tight_layout()
    plt.show()

def main():
    X_train, X_test, y_train, y_test = load_and_prepare()
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    model, preds, mae, r2 = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    save_predictions(y_test, preds)
    plot_preds(y_test, preds)

if __name__ == "__main__":
    main()
