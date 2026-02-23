# ============================================
# Project 12: Bike Rental Forecast using ARIMA
# ============================================

# ========= 1. Import Libraries =========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ========= 2. Generate Simulated Time-Series Data =========
np.random.seed(42)

# Create 2 years of daily data
dates = pd.date_range(start='2020-01-01', periods=730, freq='D')

# Simulate seasonal bike rental pattern
rentals = (
    100
    + 20 * np.sin(2 * np.pi * dates.dayofyear / 365)   # yearly seasonality
    + 5 * np.random.randn(730)                         # random noise
)

df = pd.DataFrame({'date': dates, 'rentals': rentals})
df.set_index('date', inplace=True)

print("Dataset Shape:", df.shape)
print(df.head())


# ========= 3. Visualize Time Series =========
plt.figure(figsize=(12,4))
plt.plot(df['rentals'])
plt.title("Daily Bike Rentals Over Time")
plt.xlabel("Date")
plt.ylabel("Rental Count")
plt.grid(True)
plt.show()


# ========= 4. Check Stationarity (ADF Test) =========
result = adfuller(df['rentals'])

print("\nADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("Series is Stationary ✅")
else:
    print("Series is NOT Stationary ❌ (Differencing will be used automatically)")


# ========= 5. Train-Test Split =========
train = df['rentals'][:-30]
test = df['rentals'][-30:]

print("\nTrain size:", len(train))
print("Test size:", len(test))


# ========= 6. Fit Seasonal ARIMA using Auto-ARIMA =========
model = auto_arima(
    train,
    seasonal=True,
    m=365,                 # yearly seasonality
    trace=True,
    suppress_warnings=True,
    stepwise=True
)

print("\nBest ARIMA Model:")
print(model.summary())


# ========= 7. Forecast Future Values =========
forecast = model.predict(n_periods=30)


# ========= 8. Plot Forecast vs Actual =========
plt.figure(figsize=(12,5))
plt.plot(test.index, test.values, label="Actual")
plt.plot(test.index, forecast, label="Forecast", linestyle='--')

plt.title("Bike Rentals Forecast vs Actual (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Rental Count")
plt.legend()
plt.grid(True)
plt.show()


# ========= 9. Evaluate Model =========
mse = mean_squared_error(test, forecast)
mae = mean_absolute_error(test, forecast)

print("\nModel Performance:")
print("Mean Squared Error:", round(mse, 2))
print("Mean Absolute Error:", round(mae, 2))