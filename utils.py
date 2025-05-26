import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def load_dataset(name):
    path = f"./data/{name}.csv"
    df = pd.read_csv(path)

    if name == "heart_disease":
        target_col = "target"
    elif name == "breast_cancer":
        target_col = "target"
    elif name == "parkinsons":
        target_col = "status"
    else:
        raise ValueError("Unknown dataset name")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale features and encode labels
    X = StandardScaler().fit_transform(X)
    y = LabelEncoder().fit_transform(y)

    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # Apply SMOTE to training set only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Return tensors
    return (
        torch.tensor(X_train_resampled, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train_resampled, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long)
    )
