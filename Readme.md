# Step 4: Preprocessing
from sklearn.impute import SimpleImputer

# Keep only numeric features
X_num = X.select_dtypes(include=[np.number])

# Impute missing values (replace NaN with median)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_num)

# Convert back to DataFrame with column names
X_prepared = pd.DataFrame(X_imputed, columns=X_num.columns)

print("Final feature shape:", X_prepared.shape)
X_prepared.head()
