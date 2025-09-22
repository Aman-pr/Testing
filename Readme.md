# Step 6: Feature Importance

import matplotlib.pyplot as plt

# Get feature importances
importances = rf.feature_importances_
features = X_prepared.columns

# Sort by importance
indices = importances.argsort()[::-1]

# Plot top 15 features
plt.figure(figsize=(10,6))
plt.bar(range(15), importances[indices[:15]], align="center")
plt.xticks(range(15), [features[i] for i in indices[:15]], rotation=45, ha="right")
plt.ylabel("Importance")
plt.title("Top 15 Feature Importances for Cloudburst Prediction")
plt.show()
