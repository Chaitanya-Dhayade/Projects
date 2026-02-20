# =========================
# Import Libraries
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================
# Load Dataset
# =========================
# Make sure your file is named spam.csv
df = pd.read_csv(r"G:\Projects\Naive Bayes\spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())


# =========================
# Encode Labels
# =========================
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("\nClass Distribution:\n", df['label'].value_counts())


# =========================
# Train-Test Split
# =========================
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# TF-IDF Vectorization
# =========================
vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =========================
# Train Naive Bayes Model
# =========================
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# =========================
# Prediction
# =========================
y_pred = model.predict(X_test_vec)


# =========================
# Evaluation
# =========================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))


# =========================
# Confusion Matrix Heatmap
# =========================
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix - Spam Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()