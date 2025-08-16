import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np

# Load data
data = [
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 4],
    [1, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 1, 1, 3]
]
df = pd.DataFrame(data)
X = df.iloc[:, 1:]  # Features (columns 1-6)
y = df.iloc[:, 0]   # Target (column 0)

# Random Forest Feature Importance
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
print("Feature importances:", model.feature_importances_)

# PCA Analysis
pca = PCA(n_components=2)  # Reduce to 2 principal components
X_pca = pca.fit_transform(X)

# Get PCA loadings (components)
loadings = pca.components_

# Create a DataFrame for clarity
loadings_df = pd.DataFrame(
    loadings,
    columns=[f"Feature {i+1}" for i in range(X.shape[1])],
    index=["PC1", "PC2"]
)

print("PCA Loadings (Feature Contributions):")
print(loadings_df)

print("\nPCA Results:")
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Principal Components (2D):\n", X_pca)

# Optional: Plot PCA (if you have matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Dataset (Colored by Target)')
plt.colorbar(label='Target (0 or 1)')
plt.show()

