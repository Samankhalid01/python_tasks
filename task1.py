import pandas as pd
df=pd.read_csv("./archive/iris.csv")
print(df.head()) #display first 5 rows
print(df.info())  # it displays the datatype of each column and non-null counts and memory usage
print(df.describe()) # it displays the statistical summary of numerical columns
# Select numeric columns for calculating basic stats
numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    print(f"Column: {col}")
    print(f"  Mean: {df[col].mean()}")
    print(f"  Min: {df[col].min()}")
    print(f"  Max: {df[col].max()}")
    print(f"  Count: {df[col].count()}")
    print()
#most frequent value calculation Select categorical columns
    
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    most_frequent = df[col].mode()[0]
    print(f"Column: {col}")
    print(f"  Most frequent category: {most_frequent}")
    print()
