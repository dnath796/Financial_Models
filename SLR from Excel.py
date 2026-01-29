import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0. SciPy Import with Fallback
# -----------------------------
try:
    from scipy import stats
    use_scipy = True
except ImportError:
    print("SciPy not found. Using normal approximation for p-values and CI.")
    use_scipy = False
    from math import erf, sqrt
    # Normal CDF approximation
    def norm_cdf(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

# -----------------------------
# 1. Read Excel File
# -----------------------------
# Excel file should have columns 'X' and 'Y'
df = pd.read_excel('Book1.xlsx', engine='openpyxl')
x = df['X'].values
y = df['Y'].values
n = len(x)
df_resid = n - 2  # degrees of freedom

# -----------------------------
# 2. Regression Coefficients
# -----------------------------
cov_xy = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (n - 1)
var_x = np.var(x, ddof=1)

beta = cov_xy / var_x
alpha = np.mean(y) - beta*np.mean(x)

# -----------------------------
# 3. Fitted Values & Residuals
# -----------------------------
y_hat = alpha + beta*x
residuals = y - y_hat
SSE = np.sum(residuals**2)
RSE = np.sqrt(SSE / df_resid)

# -----------------------------
# 4. Residual Variance & R²
# -----------------------------
residual_variance = np.var(residuals, ddof=1)
ss_total = np.sum((y - np.mean(y))**2)
r2 = 1 - (SSE / ss_total)

# -----------------------------
# 5. Standard Errors
# -----------------------------
SE_beta = RSE / np.sqrt(np.sum((x - np.mean(x))**2))
SE_alpha = RSE * np.sqrt(1/n + (np.mean(x)**2) / np.sum((x - np.mean(x))**2))

# -----------------------------
# 6. t-statistics
# -----------------------------
t_beta = beta / SE_beta
t_alpha = alpha / SE_alpha

# -----------------------------
# 7. p-values (two-tailed)
# -----------------------------
if use_scipy:
    p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), df=df_resid))
    p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), df=df_resid))
    # t-value for 95% CI
    t_value = stats.t.ppf(0.975, df=df_resid)
else:
    p_beta = 2 * (1 - norm_cdf(abs(t_beta)))
    p_alpha = 2 * (1 - norm_cdf(abs(t_alpha)))
    t_value = 2  # approximate for 95% CI for small n

# -----------------------------
# 8. Confidence Intervals
# -----------------------------
beta_CI = (beta - t_value*SE_beta, beta + t_value*SE_beta)
alpha_CI = (alpha - t_value*SE_alpha, alpha + t_value*SE_alpha)

# -----------------------------
# 9. Covariance & Standard Deviations
# -----------------------------
cov_xy = np.sum((x - np.mean(x))*(y - np.mean(y))) / (n-1)
std_x = np.std(x, ddof=1)
std_y = np.std(y, ddof=1)

# -----------------------------
# 10. Print Summary
# -----------------------------
print("\n--- Simple Linear Regression Analysis ---")
print(f"Regression Equation: y = {alpha:.4f} + {beta:.4f}x\n")
print(f"Covariance (X,Y): {cov_xy:.4f}")
print(f"Standard Deviation X: {std_x:.4f}")
print(f"Standard Deviation Y: {std_y:.4f}\n")
print(f"Residual Standard Error (RSE): {RSE:.4f}")
print(f"Residual Variance: {residual_variance:.4f}")
print(f"R² Score: {r2:.4f}\n")
print(f"SE_alpha: {SE_alpha:.4f}, SE_beta: {SE_beta:.4f}")
print(f"t_alpha: {t_alpha:.4f}, t_beta: {t_beta:.4f}")
print(f"p-value alpha: {p_alpha:.4f}, p-value beta: {p_beta:.4f}")
print(f"95% CI Alpha: {alpha_CI}")
print(f"95% CI Beta: {beta_CI}")

# -----------------------------
# 11. Plot Regression & Residuals
# -----------------------------
plt.figure(figsize=(12,5))

# Regression plot
plt.subplot(1,2,1)
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_hat, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Regression Fit")
plt.legend()

# Residual plot
plt.subplot(1,2,2)
plt.scatter(x, residuals, color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("X")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()
plt.show()
