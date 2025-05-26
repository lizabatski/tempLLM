import pandas as pd
import os

# Column names from UCI WDBC dataset
column_names = [
    "id", "diagnosis",  # ID and label
    # 30 features
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]


df = pd.read_csv("data_raw/wdbc.data", header=None, names=column_names)

# drop id
df = df.drop(columns=["id"])

# encode diagnosis
df["target"] = df["diagnosis"].map({"M": 1, "B": 0})
df = df.drop(columns=["diagnosis"])

# Save
os.makedirs("data", exist_ok=True)
df.to_csv("data/breast_cancer.csv", index=False)

print("Saved cleaned dataset to: data/breast_cancer.csv")
