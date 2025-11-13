# cli_data_analyzer.py

import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("=== Welcome to CLI Data Analyzer ===")

    filename = input("Enter the CSV filename write the extension too: ")

    
    try:
        df = pd.read_csv(filename)
        print(f"\nSuccessfully loaded '{filename}'!\n")
    except FileNotFoundError:
        print("File not found. Make sure the file is in the same folder as this script.")
        return

    
    print("First 5 rows of the dataset:")
    print(df.head())

 
    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nData statistics (numeric columns):")
    print(df.describe())

    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns
    print("\nNumeric columns (could be normalized):", list(numeric_cols))
    print("Categorical columns (could be encoded):", list(categorical_cols))

    while True:
        command = input("\nEnter a command (summary, head, missing, plot, exit): ").lower()

        if command == "summary":
            print("\nSummary statistics:")
            print(df.describe())
        elif command == "head":
            print("\nFirst 5 rows:")
            print(df.head())
        elif command == "missing":
            print("\nMissing values per column:")
            print(df.isna().sum())
        elif command == "plot":
            print("\nGenerating histograms for numeric columns...")
            for col in numeric_cols:
                df[col].hist()
                plt.title(col)
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.show()
        elif command == "exit":
            print("Exiting CLI Data Analyzer. Goodbye!")
main()