import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

ticker = "^GSPC"

data = yf.download(ticker, start="2018-01-01", end="2024-01-01")

# -------------------------
# 1. FLATTEN columns safely
# -------------------------
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [' '.join(col).strip() for col in data.columns.values]

print("COLUMNS AFTER FIX:", data.columns)

# -------------------------
# 2. FIND price column safely
# -------------------------
price_col = None

for col in data.columns:
    if "Close" in col:
        price_col = col
        break

if price_col is None:
    raise Exception("No Close column found. Check data output.")

# -------------------------
# 3. Compute returns
# -------------------------
prices = data[price_col]

returns = np.log(prices / prices.shift(1)).dropna()

print("\nPrice column used:", price_col)
print(returns.head())

# Portfolio value
portfolio_value = 100000  # $100K
confidence = 0.95

mean = returns.mean()
std = returns.std()

z = norm.ppf(1 - confidence)

# =========================
# 2. Parametric VaR
# =========================
parametric_var = portfolio_value * -(mean + z * std)

# =========================
# 3. Historical VaR
# =========================
hist_percentile = np.percentile(returns, (1 - confidence) * 100)
historical_var = -hist_percentile * portfolio_value

# =========================
# 4. Monte Carlo VaR
# =========================
np.random.seed(42)
simulations = 10000

sim_returns = np.random.normal(mean, std, simulations)
mc_percentile = np.percentile(sim_returns, (1 - confidence) * 100)
monte_carlo_var = -mc_percentile * portfolio_value

# =========================
# 5. Print Results
# =========================
print("\n VALUE AT RISK (VaR) RESULTS")
print("--------------------------------")
print(f"Parametric VaR  : ${parametric_var:,.2f}")
print(f"Historical VaR  : ${historical_var:,.2f}")
print(f"Monte Carlo VaR : ${monte_carlo_var:,.2f}")

# =========================
# 6. Visualization
# =========================
results = pd.DataFrame({
    "Method": ["Parametric", "Historical", "Monte Carlo"],
    "VaR": [parametric_var, historical_var, monte_carlo_var]
})

plt.figure()
plt.bar(results["Method"], results["VaR"])
plt.title("Value at Risk Comparison (95% Confidence)")
plt.ylabel("Loss ($)")
plt.show()