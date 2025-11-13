import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# -------------------- Setup Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# -------------------- Step 1: Load Dataset --------------------
def load_data(file_path):
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Dataset loaded successfully with shape: {df.shape}")
    return df

# -------------------- Step 2: Clean Data --------------------
def clean_data(df):
    logger.info("Starting data cleaning...")
    
    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna("missing", inplace=True)
    logger.info("Handled missing values.")

    # Encode categorical variables
    for col in df.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        logger.info(f"Encoded column: {col}")

    logger.info("Data cleaning complete.")
    return df

# -------------------- Step 3: Split and Scale --------------------
def preprocess_data(df, target_column):
    logger.info("Splitting data into train/test sets...")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Feature scaling complete.")

    return X_train_scaled, X_test_scaled, y_train, y_test

# -------------------- Step 4: Train Model --------------------
def train_model(X_train, y_train):
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    logger.info("Model training complete.")
    return model

# -------------------- Step 5: Generate and Save Predictions --------------------
def generate_predictions(model, X_test, y_test):
    logger.info("Generating predictions...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Model Accuracy: {accuracy:.4f}")

    output = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })
    os.makedirs("output", exist_ok=True)
    output_file = "output/predictions.csv"
    output.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")

# -------------------- Step 6: Main Workflow --------------------
def main():
    logger.info("Starting ML pipeline...")

    # Example: using Iris dataset
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["target"] = iris.target  # ensure target column name is consistent

    # Save sample dataset to CSV for demo
    sample_file = "iris.csv"
    df.to_csv(sample_file, index=False)

    data = load_data(sample_file)
    clean_df = clean_data(data)
    X_train, X_test, y_train, y_test = preprocess_data(clean_df, target_column="target")
    model = train_model(X_train, y_train)
    generate_predictions(model, X_test, y_test)

    logger.info("Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
