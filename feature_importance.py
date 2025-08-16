import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
X = df.iloc[:, 1:]  # Features (columns 1-7)
y = df.iloc[:, 0]   # Target (column 0)

# Add column names for clarity
X.columns = [f'Feature_{i}' for i in range(1, 8)]

# 1. Correlation Matrix
corr_matrix = X.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Feature Correlation Matrix")
plt.show()

# 2. Random Forest Feature Importance
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
print("\nFeature importances:", model.feature_importances_)

# 3. PCA Analysis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Get PCA loadings (components)
loadings = pca.components_
loadings_df = pd.DataFrame(
    loadings,
    columns=X.columns,
    index=["PC1", "PC2"]
)

print("\nPCA Loadings (Feature Contributions):")
print(loadings_df)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

# Plot PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Dataset (Colored by Target)')
plt.colorbar(label='Target (0 or 1)')
plt.show()