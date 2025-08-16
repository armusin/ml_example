import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

# Load data
data = [
    [0,0,1,1,0,0],
    [0,0,0,0,1,1],
    [0,1,1,1,1,1],
    [0,1,1,1,1,1],
    [1,1,0,1,0,0],
    [1,1,1,0,1,1],
    [1,1,0,1,1,1]
]

df = pd.DataFrame(data)
X = df.iloc[:, 1:]  # Features (columns 1-5)
y = df.iloc[:, 0]   # Target (column 0)

# Initialize models
log_reg = LogisticRegression(max_iter=1000)
dtree = DecisionTreeClassifier(max_depth=2)

# Leave-One-Out Cross-Validation
loo = LeaveOneOut()

# Evaluate Logistic Regression
log_scores = cross_val_score(log_reg, X, y, cv=loo)
print("Logistic Regression Accuracy:", log_scores.mean())

# Evaluate Decision Tree
dt_scores = cross_val_score(dtree, X, y, cv=loo)
print("Decision Tree Accuracy:", dt_scores.mean())
