# =========================
# Import Libraries
# =========================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# Load Dataset
# =========================
digits = load_digits()

print("Image Data Shape:", digits.images.shape)

# Visualize first 5 images
plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")
    plt.axis('off')
plt.suptitle("Sample Handwritten Digits")
plt.show()


# =========================
# Prepare Data
# =========================
X = digits.data        # Flattened (1797, 64)
y = digits.target      # Labels 0–9


# Scale features (important for SVM)
scaler = StandardScaler()
X = scaler.fit_transform(X)


# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# Train SVM Model
# =========================
model = SVC(
    kernel='rbf',
    gamma=0.001,
    C=10
)

model.fit(X_train, y_train)


# =========================
# Evaluate Model
# =========================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =========================
# Confusion Matrix
# =========================
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix - Digit Recognition")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# =========================
# Show Predictions
# =========================
plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_test[i].reshape(8,8), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}")
    plt.axis('off')
plt.suptitle("SVM Predictions on Test Images")
plt.show()