import pandas as pd
import numpy as np

# Load the data to check dtypes
df = pd.read_csv('./data/processed/train_features.csv')
print("Data shape:", df.shape)
print("\nColumn dtypes:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())
print("\nData types of first row:")
print(df.iloc[0].apply(type))

# Check for object columns
print("\nObject columns:")
object_cols = df.select_dtypes(include=['object']).columns
print(object_cols)

# Convert boolean columns to int
for col in df.select_dtypes(include=['bool']).columns:
    df[col] = df[col].astype(int)

print("\nAfter converting booleans to int:")
print(df.dtypes)
print("\nSample of converted data:")
print(df.head())