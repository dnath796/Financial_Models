import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Read Excel File
# -----------------------------
# Replace 'data.xlsx' with your Excel file path
# Make sure columns are named 'X' and 'Y'
df = pd.read_excel('Book1.xlsx', engine='openpyxl')

x = df['X'].values
y = df['Y'].values
n = len(x)

# -----------------------------
# 2. Regression Coefficients (Alpha & Beta)
# -----------------------------
cov_xy = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (n - 1)
var_x = np.var(x, ddof=1)

beta = cov_xy / var_x
alpha = np.mean(y) - beta*np.mean(x)

print(f"Regression Equation: y = {alpha:.2f} + {beta:.2f}x")

# -----------------------------
# 3. Fitted Values and Residuals
# -----------------------------
y_hat = alpha + beta*x
residuals = y - y_hat

print("Fitted Values (y_hat):", y_hat)
print("Residuals:", residuals)

# -----------------------------
# 4. Residual Standard Error (RSE)
# -----------------------------
SSE = np.sum(residuals**2)
RSE = np.sqrt(SSE / (n - 2))
print("Residual Standard Error (RSE):", RSE)

# -----------------------------
# 5. Variance of Residuals
# -----------------------------
residual_variance = np.var(residuals, ddof=1)
print("Variance of Residuals:", residual_variance)

# -----------------------------
# 6. R² (Goodness of Fit)
# -----------------------------
ss_total = np.sum((y - np.mean(y))**2)
r2 = 1 - (SSE / ss_total)
print("R² Score:", r2)

# -----------------------------
# 7. Standard Errors for Alpha and Beta
# -----------------------------
SE_beta = RSE / np.sqrt(np.sum((x - np.mean(x))**2))
SE_alpha = RSE * np.sqrt(1/n + (np.mean(x)**2) / np.sum((x - np.mean(x))**2))

# -----------------------------
# 8. Confidence Intervals (95%) Without SciPy
# -----------------------------
# t-value approximation for small n (use proper table if you want)
# df = n - 2
# For n=5, df=3, t ≈ 3.182 (95% CI)
t_value = 3.182

beta_CI = (beta - t_value*SE_beta, beta + t_value*SE_beta)
alpha_CI = (alpha - t_value*SE_alpha, alpha + t_value*SE_alpha)

print(f"95% CI for Beta (slope): {beta_CI}")
print(f"95% CI for Alpha (intercept): {alpha_CI}")

# -----------------------------
# 9. Covariance and Standard Deviations
# -----------------------------
cov_xy = np.sum((x - np.mean(x))*(y - np.mean(y)))/(n-1)
std_x = np.std(x, ddof=1)
std_y = np.std(y, ddof=1)
print("Covariance (X,Y):", cov_xy)
print("Standard Deviation X:", std_x)
print("Standard Deviation Y:", std_y)

# -----------------------------
# 10. Plot Regression & Residuals
# -----------------------------
plt.figure(figsize=(12,5))

# Regression plot
plt.subplot(1,2,1)
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_hat, color='red', label='Regression Line')
plt.xlabel("X (Independent Variable)")
plt.ylabel("Y (Dependent Variable)")
plt.title("Regression Fit")
plt.legend()

# Residual plot
plt.subplot(1,2,2)
plt.scatter(x, residuals, color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("X (Independent Variable)")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()
plt.show()
