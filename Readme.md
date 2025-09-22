# Step 3: Data Cleaning & Target Preparation

import numpy as np

# Define a helper function to convert Yes/No to binary
def to_binary(series):
    s = series.astype(str).str.strip().str.lower()
    return s.map(lambda x: 1 if x in ["yes","y","1","true","t"] 
                 else 0 if x in ["no","n","0","false","f"] 
                 else np.nan)

# Select target column (label)
target_col = "CloudBurst Tomorrow"  # <-- change if different
df["target"] = to_binary(df[target_col])

# Drop rows where target is missing
df = df[df["target"].notna()].copy()

print("Target distribution (0 = No, 1 = Yes):")
print(df["target"].value_counts())

# Drop target column + any ID columns from features
X = df.drop(columns=[target_col, "target"])
y = df["target"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)
