# For data manipulation
import pandas as pd
import numpy as np
import os

# For data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Hugging Face
from huggingface_hub import HfApi, hf_hub_download

# Authenticate
api = HfApi(token=os.getenv("HF_TOKEN"))

# Download dataset from Hugging Face
csv_path = hf_hub_download(
    repo_id="ShanRaja/Customer-Purchase-Prediction",
    repo_type="dataset",
    filename="tourism.csv"
)

df = pd.read_csv(csv_path)
print("Dataset loaded successfully.")

# Preprocessing
df.drop(columns=['CustomerID'], inplace=True)
df['CityTier'] = df['CityTier'].astype(int)

X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Columns for encoding
binary_cols = ['Gender']
onehot_cols = ['TypeofContact','Occupation','MaritalStatus','ProductPitched','Designation']

# Label encode binary column
le = LabelEncoder()
for col in binary_cols:
    X_train[col] = le.fit_transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    X_test[col] = le.transform(X_test[col])

# One-hot encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols)],
    remainder='passthrough'
)

# Save preprocessed data
X_train.to_csv("X_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Upload to Hugging Face Dataset repo

files = [
    "X_train.csv", "X_val.csv", "X_test.csv",
    "y_train.csv", "y_val.csv", "y_test.csv"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="ShanRaja/Customer-Purchase-Prediction",
        repo_type="dataset",
    )

print("All files uploaded successfully!")
