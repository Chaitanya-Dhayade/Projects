# =========================
# Import Libraries
# =========================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# =========================
# Load Stock Data
# =========================
df = yf.download("AAPL", start="2018-01-01", end="2023-01-01")

data = df[['Close']].copy()
data.dropna(inplace=True)

plt.figure(figsize=(10,4))
plt.plot(data)
plt.title("AAPL Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()


# =========================
# Normalize Data
# =========================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# =========================
# Create Sequences (60 days → next day)
# =========================
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM → (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================
# Train-Test Split (80-20)
# =========================
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# =========================
# Build LSTM Model
# =========================
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))  # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')


# =========================
# Train Model
# =========================
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)


# =========================
# Predict Prices
# =========================
predictions = model.predict(X_test)

predicted_prices = scaler.inverse_transform(predictions.reshape(-1,1))
actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))


# =========================
# Plot Predictions
# =========================
plt.figure(figsize=(10,5))
plt.plot(actual_prices, label="Actual Price")
plt.plot(predicted_prices, label="Predicted Price")
plt.title("AAPL Stock Price Prediction (LSTM)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()