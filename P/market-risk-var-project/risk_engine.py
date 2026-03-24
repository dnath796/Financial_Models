import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from arch import arch_model

# =========================
# 1. LOAD DATA
# =========================
data = yf.download("^GSPC", start="2018-01-01", auto_adjust=True)

prices = data["Close"].squeeze()
returns = np.log(prices).diff().dropna()

portfolio_value = 100000
confidence = 0.95

mean = returns.mean()
std = returns.std()
z = norm.ppf(1 - confidence)

# =========================
# 2. VaR METHODS
# =========================

# Parametric VaR
parametric_var = -portfolio_value * (mean + z * std)

# Historical VaR
hist_var = -portfolio_value * np.percentile(returns, (1 - confidence) * 100)

# Monte Carlo VaR
np.random.seed(42)
sim_returns = np.random.normal(mean, std, 10000)
mc_var = -portfolio_value * np.percentile(sim_returns, (1 - confidence) * 100)

# =========================
# 3. EXPECTED SHORTFALL (ES)
# =========================
var_threshold = np.percentile(returns, (1 - confidence) * 100)
es = -portfolio_value * returns[returns <= var_threshold].mean()

# =========================
# 4. GARCH MODEL
# =========================
garch = arch_model(returns * 100, vol='Garch', p=1, q=1)
garch_fit = garch.fit(disp="off")

garch_vol = garch_fit.conditional_volatility / 100

# GARCH-based VaR
garch_var = -portfolio_value * (mean + z * garch_vol.iloc[-1])

# =========================
# 5. STRESS TESTING
# =========================
stress_scenarios = {
    "Market Crash (-20%)": -0.20,
    "Severe Crash (-30%)": -0.30,
    "Mild Shock (-10%)": -0.10
}

stress_results = {k: v * portfolio_value for k, v in stress_scenarios.items()}

# =========================
# 6. BACKTESTING (Kupiec)
# =========================
var_series = - (mean + z * std)

violations = returns < var_series
num_violations = violations.sum()
total_obs = len(returns)

violation_ratio = num_violations / total_obs

# =========================
# 7. RESULTS
# =========================
print("\n📊 VALUE AT RISK")
print("------------------------")
print(f"Parametric VaR : ${parametric_var:,.2f}")
print(f"Historical VaR : ${hist_var:,.2f}")
print(f"Monte Carlo VaR: ${mc_var:,.2f}")
print(f"GARCH VaR      : ${garch_var:,.2f}")

print("\n📉 EXPECTED SHORTFALL")
print("------------------------")
print(f"ES (CVaR)      : ${es:,.2f}")

print("\n🔥 STRESS TESTING")
print("------------------------")
for k, v in stress_results.items():
    print(f"{k}: ${v:,.2f}")

print("\n📊 BACKTESTING")
print("------------------------")
print(f"Violations     : {num_violations}")
print(f"Violation Rate : {violation_ratio:.4f}")