import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Sample Data
# -----------------------------
x = np.array([1, 2, 3, 4, 5])  # Independent variable
y = np.array([2, 4, 5, 4, 5])  # Dependent variable

# -----------------------------
# 2. Calculate Slope (m) and Intercept (b)
# -----------------------------
n = len(x)
m = (n * np.sum(x*y) - np.sum(x)*np.sum(y)) / (n*np.sum(x**2) - (np.sum(x))**2)
b = (np.sum(y) - m*np.sum(x)) / n

print("Slope (m):", m)
print("Intercept (b):", b)
print(f"Regression Equation: y = {m:.2f}x + {b:.2f}")

# -----------------------------
# 3. Make Predictions
# -----------------------------
y_pred = m*x + b
print("Predicted values:", y_pred)

# Predict for a new value
new_x = 6
new_pred = m*new_x + b
print(f"Prediction for X = {new_x}: {new_pred:.2f}")

# -----------------------------
# 4. Calculate R² Score
# -----------------------------
ss_total = np.sum((y - np.mean(y))**2)
ss_res = np.sum((y - y_pred)**2)
r2 = 1 - (ss_res / ss_total)
print("R² Score:", r2)

# -----------------------------
# 5. Plot Results
# -----------------------------
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.xlabel("X (Independent Variable)")
plt.ylabel("y (Dependent Variable)")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
