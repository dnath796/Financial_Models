# S&P 500 Volatility Modeling and Non-Linearity Testing
# Author: Your Name
# Description: End-to-end project for testing non-linearity and modeling volatility using GARCH

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset, het_arch
from arch import arch_model

# -----------------------------
# 1. Load Data
# -----------------------------
data = yf.download("^GSPC", start="2015-01-01", end="2024-01-01")

# -----------------------------
# 2. Compute Log Returns
# -----------------------------
data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
data = data.dropna()

# -----------------------------
# 3. Plot Price and Returns
# -----------------------------
plt.figure()
plt.plot(data['Close'])
plt.title("S&P 500 Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

plt.figure()
plt.plot(data['returns'])
plt.title("S&P 500 Log Returns")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.show()

# -----------------------------
# 4. Linear Model + RESET Test
# -----------------------------
X = sm.add_constant(np.arange(len(data)))
model = sm.OLS(data['returns'], X).fit()
reset_test = linear_reset(model, power=2, use_f=True)
print("RESET Test p-value:", reset_test.pvalue)

# -----------------------------
# 5. ARCH Test (Non-linearity in volatility)
# -----------------------------
arch_test = het_arch(data['returns'])
print("ARCH Test p-value:", arch_test[1])

# -----------------------------
# 6. Fit GARCH Model
# -----------------------------
garch = arch_model(data['returns'] * 100, vol='Garch', p=1, q=1)
garch_fit = garch.fit()
print(garch_fit.summary())

# -----------------------------
# 7. Plot Conditional Volatility
# -----------------------------
plt.figure()
plt.plot(garch_fit.conditional_volatility)
plt.title("GARCH Conditional Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.show()

# -----------------------------
# 8. Save Results
# -----------------------------
data.to_csv("sp500_returns.csv")

# -----------------------------
# 9. BDS Test (Non-linear Dependence)
# -----------------------------
from statsmodels.tsa.stattools import bds

# BDS test on returns
bds_stat = bds(data['returns'])
print("BDS Test Statistic:", bds_stat[0])
print("BDS Test p-value:", bds_stat[1])

# -----------------------------
# 10. Linear vs GARCH Comparison
# -----------------------------
from sklearn.metrics import mean_squared_error

# Linear model predictions (mean only)
linear_pred = model.fittedvalues

# GARCH volatility (proxy for risk prediction)
garch_vol = garch_fit.conditional_volatility

# Compare variance (MSE on squared returns)
actual_vol = (data['returns'] * 100) ** 2

linear_mse = mean_squared_error(actual_vol, linear_pred**2)
garch_mse = mean_squared_error(actual_vol, garch_vol**2)

print("Model Comparison:")
print("Linear Model MSE:", linear_mse)
print("GARCH Model MSE:", garch_mse)

# -----------------------------
# 11. Plot Comparison
# -----------------------------
plt.figure()
plt.plot(actual_vol, label='Actual Volatility (Squared Returns)')
plt.plot(garch_vol**2, label='GARCH Estimated Volatility')
plt.title("Volatility: Actual vs GARCH")
plt.legend()
plt.show()

print("Project Completed Successfully!")
