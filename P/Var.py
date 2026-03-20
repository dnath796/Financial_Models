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

bds_stat = bds(data['returns'])
print("BDS Test Statistic:", bds_stat[0])
print("BDS Test p-value:", bds_stat[1])

# -----------------------------
# 10. Linear vs GARCH Comparison
# -----------------------------
from sklearn.metrics import mean_squared_error

linear_pred = model.fittedvalues
garch_vol = garch_fit.conditional_volatility
actual_vol = (data['returns'] * 100) ** 2

linear_mse = mean_squared_error(actual_vol, linear_pred**2)
garch_mse = mean_squared_error(actual_vol, garch_vol**2)

print("Model Comparison:")
print("Linear Model MSE:", linear_mse)
print("GARCH Model MSE:", garch_mse)

plt.figure()
plt.plot(actual_vol, label='Actual Volatility (Squared Returns)')
plt.plot(garch_vol**2, label='GARCH Estimated Volatility')
plt.title("Volatility: Actual vs GARCH")
plt.legend()
plt.show()

# -----------------------------
# 11. Value at Risk (VaR) Backtesting
# -----------------------------
import scipy.stats as stats

confidence_level = 0.95
z_score = stats.norm.ppf(1 - confidence_level)

garch_vol = garch_fit.conditional_volatility / 100
VaR = z_score * garch_vol

returns = data['returns']
violations = returns < VaR

num_violations = violations.sum()
total_obs = len(returns)
violation_ratio = num_violations / total_obs

print("VaR Backtesting Results:")
print("Total Observations:", total_obs)
print("Number of Violations:", num_violations)
print("Violation Ratio:", violation_ratio)

plt.figure()
plt.plot(returns, label='Returns')
plt.plot(VaR, label='VaR (95%)', linestyle='--')
plt.title("VaR Backtesting: Returns vs VaR")
plt.legend()
plt.show()

# -----------------------------
# 12. Kupiec Test (Proportion of Failures)
# -----------------------------
expected_ratio = 1 - confidence_level

LR = -2 * (
    num_violations * np.log(expected_ratio / violation_ratio) +
    (total_obs - num_violations) * np.log((1 - expected_ratio) / (1 - violation_ratio))
)

p_value_kupiec = 1 - stats.chi2.cdf(LR, df=1)

print("Kupiec Test:")
print("LR Statistic:", LR)
print("p-value:", p_value_kupiec)

print("Project Completed Successfully!")

# -----------------------------
# 13. Expected Shortfall (CVaR)
# -----------------------------
# CVaR = average loss beyond VaR
cvar = returns[returns < VaR].mean()

print("\nExpected Shortfall (CVaR):", cvar)

# -----------------------------
# 14. Rolling VaR Forecast (Out-of-Sample)
# -----------------------------
window = 500
rolling_var = []

for i in range(window, len(returns)):
    train = returns[i-window:i]
    model_roll = arch_model(train * 100, vol='Garch', p=1, q=1)
    fit_roll = model_roll.fit(disp='off')
    forecast = fit_roll.forecast(horizon=1)
    vol_forecast = np.sqrt(forecast.variance.values[-1, :][0]) / 100
    var_forecast = z_score * vol_forecast
    rolling_var.append(var_forecast)

rolling_var = pd.Series(rolling_var, index=returns.index[window:])

plt.figure()
plt.plot(returns[window:], label='Returns')
plt.plot(rolling_var, label='Rolling VaR', linestyle='--')
plt.title("Rolling VaR Forecast")
plt.legend()
plt.show()

# -----------------------------
# 15. Basel Traffic Light Test
# -----------------------------
violations_rolling = returns[window:] < rolling_var
num_violations_rolling = violations_rolling.sum()

print("\nBasel Traffic Light Test:")
print("Violations (last window):", num_violations_rolling)

if num_violations_rolling <= 4:
    print("Green Zone: Model is acceptable")
elif num_violations_rolling <= 9:
    print("Yellow Zone: Model needs improvement")
else:
    print("Red Zone: Model is not acceptable")

print("\nProject Completed Successfully!")
