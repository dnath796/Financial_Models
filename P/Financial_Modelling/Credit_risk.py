import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# PARAMETERS
# ============================
np.random.seed(42)

n_paths = 10000        # number of Monte Carlo simulations
n_steps = 50           # time steps
T = 1.0                # 1 year
dt = T / n_steps

S0 = 100               # initial price
mu = 0.05              # drift
sigma = 0.2            # volatility
K = 100                # strike (for option-like exposure)

# ============================
# SIMULATE GBM PATHS
# ============================
S = np.zeros((n_paths, n_steps + 1))
S[:, 0] = S0

for t in range(1, n_steps + 1):
    Z = np.random.normal(size=n_paths)
    S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# ============================
# PORTFOLIO VALUE (Example: Call Option Exposure)
# ============================
# Exposure = max(S - K, 0)
exposure = np.maximum(S - K, 0)

# ============================
# EE (Expected Exposure)
# ============================
EE = np.mean(exposure, axis=0)

# ============================
# PFE (95th percentile exposure)
# ============================
PFE_95 = np.percentile(exposure, 95, axis=0)

# ============================
# TIME GRID
# ============================
time_grid = np.linspace(0, T, n_steps + 1)

# ============================
# PLOT RESULTS
# ============================
plt.figure(figsize=(10,6))
plt.plot(time_grid, EE, label="Expected Exposure (EE)")
plt.plot(time_grid, PFE_95, label="PFE (95%)", linestyle='--')
plt.xlabel("Time")
plt.ylabel("Exposure")
plt.title("Monte Carlo Simulation of EE and PFE")
plt.legend()
plt.grid()
plt.show()



np.random.seed(42)

# ============================
# PARAMETERS
# ============================
n_paths = 10000
n_steps = 50
T = 1.0
dt = T / n_steps

# Initial prices
S0 = np.array([100, 120, 80])

# Drifts and volatilities
mu = np.array([0.05, 0.04, 0.06])
sigma = np.array([0.2, 0.25, 0.3])

# Correlation matrix
corr = np.array([
    [1.0, 0.6, 0.3],
    [0.6, 1.0, 0.5],
    [0.3, 0.5, 1.0]
])

# Cholesky decomposition
L = np.linalg.cholesky(corr)

n_assets = len(S0)

# ============================
# SIMULATION
# ============================
S = np.zeros((n_paths, n_steps + 1, n_assets))
S[:, 0, :] = S0

for t in range(1, n_steps + 1):
    Z = np.random.normal(size=(n_paths, n_assets))
    
    # Introduce correlation
    correlated_Z = Z @ L.T
    
    for i in range(n_assets):
        S[:, t, i] = S[:, t-1, i] * np.exp(
            (mu[i] - 0.5 * sigma[i]**2) * dt +
            sigma[i] * np.sqrt(dt) * correlated_Z[:, i]
        )

# ============================
# PORTFOLIO VALUE
# ============================
# Example: long positions in all assets
weights = np.array([0.5, 0.3, 0.2])

portfolio_value = np.sum(S * weights, axis=2)

# Exposure = max(portfolio value, 0)
exposure = np.maximum(portfolio_value, 0)

# ============================
# EE & PFE
# ============================
EE = np.mean(exposure, axis=0)
PFE_95 = np.percentile(exposure, 95, axis=0)

time_grid = np.linspace(0, T, n_steps + 1)

# ============================
# PLOT
# ============================
plt.figure()
plt.plot(time_grid, EE, label="EE")
plt.plot(time_grid, PFE_95, linestyle='--', label="PFE (95%)")
plt.xlabel("Time")
plt.ylabel("Exposure")
plt.title("Multi-Asset EE & PFE")
plt.legend()
plt.grid()
plt.show()