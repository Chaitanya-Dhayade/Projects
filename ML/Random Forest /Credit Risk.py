# =========================
# Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# =========================
# Load Dataset (UCI ID = 144)
# =========================
data = fetch_ucirepo(id=144)

X = data.data.features.copy()
y = data.data.targets.copy()

# Convert target to 1D
y = y.iloc[:, 0]

print("Dataset Shape:", X.shape)
print("Target Distribution:\n", y.value_counts())


# =========================
# Encode Categorical Features
# =========================
le = LabelEncoder()

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

# Encode target (good=0, bad=1 typically)
y = le.fit_transform(y)


# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# =========================
# Train Random Forest Model
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# Predictions
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# =========================
# Evaluation
# =========================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))


# =========================
# Confusion Matrix Heatmap
# =========================
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# =========================
# ROC Curve
# =========================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="Random Forest")
plt.plot([0,1], [0,1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# =========================
# Feature Importance
# =========================
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x=importance.values[:10], y=importance.index[:10])
plt.title("Top 10 Important Features")
plt.show()
