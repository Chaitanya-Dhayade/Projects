# =========================
# Import necessary libraries
# =========================
import pandas as pd              # Data handling
import numpy as np               # Numerical operations
import matplotlib.pyplot as plt # Plotting
import seaborn as sns            # Statistical plotting

from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.linear_model import LinearRegression     # ML model
from sklearn.metrics import mean_squared_error, r2_score  # Evaluation metrics
from sklearn.datasets import load_boston              # Dataset (deprecated)


# =========================
# Load and prepare dataset
# =========================
boston = load_boston()

# Convert to pandas DataFrame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target   # Target variable (median house price)

# Display first few rows
print(df.head())


# =========================
# Correlation Heatmap
# =========================
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Boston Housing Data")
plt.show()


# =========================
# Select feature and target
# =========================
X = df[['RM']]      # Feature: average number of rooms
y = df['PRICE']     # Target: median house price


# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# Train Linear Regression model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Coefficient (slope):", model.coef_)
print("Model Intercept (bias):", model.intercept_)


# =========================
# Predictions and Evaluation
# =========================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)


# =========================
# Plot regression line
# =========================
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')

plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Median House Price ($1000s)')
plt.title('Linear Regression: RM vs PRICE')
plt.legend()
plt.show()
