import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from arch import arch_model

# -----------------------------
# Load Data (S&P 500)
# -----------------------------
data = yf.download("^GSPC", start="2015-01-01", end="2025-01-01", auto_adjust=True)
price = data["Close"]
returns = price.pct_change().dropna()

portfolio_value = 1_000_000
alpha = 0.05  # 95% VaR

# -----------------------------
# Historical VaR / DEAR
# -----------------------------
hist_var = np.percentile(returns, alpha * 100)
hist_dear = hist_var * portfolio_value

# -----------------------------
# Parametric VaR / DEAR
# -----------------------------
mu = returns.mean().item()
sigma = returns.std().item()
z = norm.ppf(alpha)

param_var = mu + z * sigma
param_dear = float(param_var) * portfolio_value

# -----------------------------
# Monte Carlo VaR / DEAR
# -----------------------------
simulations = 10000
mc_returns = np.random.normal(mu, sigma, simulations)

mc_var = np.percentile(mc_returns, alpha * 100)
mc_dear = mc_var * portfolio_value

# -----------------------------
# GARCH Monte Carlo VaR / DEAR
# -----------------------------
garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1)
garch_res = garch_model.fit(disp='off')

forecast = garch_res.forecast(horizon=1)
vol = np.sqrt(forecast.variance.values[-1][0]) / 100

garch_sim = np.random.normal(0, vol, simulations)
garch_var = np.percentile(garch_sim, alpha * 100)
garch_dear = garch_var * portfolio_value

# -----------------------------
# Backtesting (Historical VaR)
# -----------------------------
violations = (returns < hist_var).sum()
expected = len(returns) * alpha

# -----------------------------
# Rolling VaR
# -----------------------------
rolling_var = returns.rolling(100).quantile(alpha)

# -----------------------------
# Results
# -----------------------------
print("\n📊 MARKET RISK RESULTS (DEAR)")
print("-" * 40)
print(f"Historical DEAR : ${hist_dear:,.2f}")
print(f"Parametric DEAR : ${param_dear:,.2f}")
print(f"Monte Carlo DEAR: ${mc_dear:,.2f}")
print(f"GARCH MC DEAR   : ${garch_dear:,.2f}")

print("\n📉 BACKTESTING (Historical VaR)")
print("-" * 40)
print(f"Violations: {violations}")
print(f"Expected  : {expected:.0f}")

# -----------------------------
# PLOTS
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(returns.values, label="Returns")
plt.plot(rolling_var.values, label="Rolling VaR (5%)", color="red")
plt.title("Rolling VaR vs S&P 500 Returns")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.hist(returns, bins=100, alpha=0.5, label="Historical")
plt.hist(mc_returns, bins=100, alpha=0.5, label="Monte Carlo")
plt.legend()
plt.title("Return Distribution Comparison")
plt.show()