import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = [
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1]
]

df = pd.DataFrame(data)
X = df.iloc[:, 1:]  # Features (columns 1-5)
y = df.iloc[:, 0]   # Target (column 0)

model = RandomForestClassifier()
model.fit(X, y)
print("Feature importances:", model.feature_importances_)
