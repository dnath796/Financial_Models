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

print("\nProject Completed Successfully!")
