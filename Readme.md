# Step 5: Train/Test Split + Train Model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_prepared, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, " Test size:", X_test.shape)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200, 
    class_weight="balanced", 
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate on test set
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC AUC:", roc_auc_score(y_test, y_proba))
