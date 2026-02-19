# =========================
# Import Libraries
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =========================
# Load Dataset
# =========================
df = pd.read_csv("G:\Ai_Projects\K-means\Mall_Customers.csv")  # Make sure file is in same folder
print(df.head())


# =========================
# Data Cleaning
# =========================
# Drop CustomerID (not useful for clustering)
df = df.drop('CustomerID', axis=1)

# Convert Gender to numeric
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

print(df.info())


# =========================
# Select Features for Clustering
# =========================
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


# =========================
# Feature Scaling (IMPORTANT)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================
# Elbow Method to Find Optimal K
# =========================
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


# =========================
# Apply K-Means (Choose k=5)
# =========================
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(df.head())


# =========================
# Visualize Clusters
# =========================
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='tab10'
)

plt.title('Customer Segments by K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()