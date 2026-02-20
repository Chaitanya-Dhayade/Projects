# =========================
# Import libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# =========================
# Load dataset
# =========================
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

print(df.head())


# =========================
# Correlation Heatmap (optional)
# =========================
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# =========================
# Select Multiple Strong Features
# =========================
#X = df[['MedInc', 'AveRooms', 'HouseAge', 'AveOccup', 'Population']]

X = df[['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']]


y = df['PRICE']


# =========================
# Feature Scaling (improves model stability)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# =========================
# Train Multiple Linear Regression
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)


# =========================
# Prediction & Evaluation
# =========================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nImproved Model Results")
print("Mean Squared Error:", mse)
print("R² Score:", r2)


# =========================
# Residual Plot (Error Visualization)
# =========================
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
