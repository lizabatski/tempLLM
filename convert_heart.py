import pandas as pd
import os


df = pd.read_csv("data_raw/processed.cleveland.data", header=None)


df.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

df = df.replace("?", pd.NA)
df = df.dropna()
df = df.apply(pd.to_numeric)


os.makedirs("data", exist_ok=True)
df.to_csv("data/heart_disease.csv", index=False)

print("Cleaned heart disease data saved to data/heart_disease.csv")
