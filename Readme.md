# Step 6: Feature Importance

import matplotlib.pyplot as plt
import numpy as np

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]   # sort descending
features = X_train.columns

# Plot top 15 features
plt.figure(figsize=(10,6))
plt.bar(range(15), importances[indices][:15], align="center")
plt.xticks(range(15), [features[i] for i in indices[:15]], rotation=45, ha="right")
plt.title("Top 15 Important Features for Cloudburst Prediction")
plt.ylabel("Importance Score")
plt.show()
