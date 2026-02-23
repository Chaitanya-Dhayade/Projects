# =========================
# Import Libraries
# =========================
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Load Dataset
# =========================
digits = load_digits()
X = digits.data      # (1797, 64)
y = digits.target    # Labels 0–9

print("Original shape of data:", X.shape)


# =========================
# Apply PCA (64 → 2)
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Reduced shape of data:", X_pca.shape)


# =========================
# Visualize in 2D
# =========================
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Label'] = y

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='Label',
    palette='tab10',
    s=60
)

plt.title('PCA Projection of Digits Dataset (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Digit')
plt.grid(True)
plt.show()


# =========================
# Explained Variance (2 Components)
# =========================
print("Explained variance ratio (PC1, PC2):")
print(pca.explained_variance_ratio_)


# =========================
# Cumulative Explained Variance
# =========================
pca_full = PCA().fit(X)
cumulative_variance = pca_full.explained_variance_ratio_.cumsum()

plt.figure(figsize=(8,4))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.axhline(y=0.95, linestyle='--')   # 95% variance line
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()