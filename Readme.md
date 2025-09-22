# Step 7: Feature Importances (which variables influence prediction most)

import pandas as pd

# Get feature importances from the trained Random Forest
importances = rf.feature_importances_

# Create a DataFrame for better viewing
feat_imp = pd.DataFrame({
    "Feature": X_prepared.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Show top 10 important features
print("Top 10 Features influencing cloudburst prediction:")
display(feat_imp.head(10))

# Plot for visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.barh(feat_imp["Feature"].head(10), feat_imp["Importance"].head(10))
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("Top 10 Important Features for Cloudburst Prediction")
plt.show()
