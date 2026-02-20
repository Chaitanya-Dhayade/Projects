# =========================
# Import Libraries
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================
# Load Dataset from UCI
# =========================
data = fetch_ucirepo(id=144)

X = data.data.features.copy()
y = data.data.targets


# Convert target to Series (if dataframe)
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

# Encode target
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
# Train Decision Tree
# =========================
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=7,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# Evaluate
# =========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# =========================
# Feature Importance
# =========================
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).head(10).plot(kind='barh', figsize=(8,6))
plt.title("Top Feature Importance")
plt.show()


# =========================
# Visualize Tree
# =========================
plt.figure(figsize=(16,8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Bad Credit','Good Credit'],
    filled=True,
    max_depth=3
)
plt.show()