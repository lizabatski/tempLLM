import pandas as pd
import os

df = pd.read_csv("data_raw/parkinsons.data")

df = df.drop(columns=["name"])


os.makedirs("data", exist_ok=True)
df.to_csv("data/parkinsons.csv", index=False)

print("Saved cleaned dataset to: data/parkinsons.csv")
