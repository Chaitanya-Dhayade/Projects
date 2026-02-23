# =========================
# Import Libraries
# =========================
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Generate Synthetic Data
# =========================
# y = 3x + 4 + noise
np.random.seed(42)

X = 2 * np.random.rand(100, 1)          # 100 points in [0,2]
y = 3 * X + 4 + np.random.randn(100, 1) # add Gaussian noise

plt.scatter(X, y)
plt.title("Synthetic Linear Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


# =========================
# Initialize Parameters
# =========================
m = 0.0   # slope
b = 0.0   # intercept

learning_rate = 0.05
epochs = 1000
n = len(X)

loss_history = []


# =========================
# Gradient Descent Loop
# =========================
for i in range(epochs):

    # Predictions
    y_pred = m * X + b

    # Error
    error = y_pred - y

    # Gradients (derivatives of MSE)
    dm = (2/n) * np.sum(X * error)
    db = (2/n) * np.sum(error)

    # Update parameters
    m -= learning_rate * dm
    b -= learning_rate * db

    # Compute loss (MSE)
    loss = (1/n) * np.sum(error**2)
    loss_history.append(loss)

    if i % 100 == 0:
        print(f"Epoch {i}: Loss={loss:.4f}, m={m:.4f}, b={b:.4f}")


# =========================
# Final Fitted Line
# =========================
plt.scatter(X, y, label="Data")
plt.plot(X, m * X + b, color="red", label="Fitted Line")
plt.title("Linear Regression via Gradient Descent")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()


# =========================
# Loss Curve
# =========================
plt.plot(range(epochs), loss_history)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()