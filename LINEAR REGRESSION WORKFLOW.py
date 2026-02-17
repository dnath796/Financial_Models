# ==============================
# COMPLETE LINEAR REGRESSION WORKFLOW
# ==============================

# Install packages if needed:
# pip install pandas numpy statsmodels scipy matplotlib

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as diag
import statsmodels.stats.stattools as stattools
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

# ==============================
# 1. LOAD DATA
# ==============================

# Replace with your dataset
df = pd.read_csv("financial_data.xlsx")

# Example column names — CHANGE if needed
DEPENDENT_VAR = 'stock_return'
INDEPENDENT_VARS = ['market_return', 'interest_rate', 'inflation']

Y = df[DEPENDENT_VAR]
X = df[INDEPENDENT_VARS]

# Add intercept
X = sm.add_constant(X)

# ==============================
# 2. RUN OLS REGRESSION
# ==============================

model = sm.OLS(Y, X).fit()
print("\n===== OLS REGRESSION RESULTS =====")
print(model.summary())

residuals = model.resid
fitted = model.fittedvalues
T = len(Y)
k = X.shape[1]

# ==============================
# 3. HETEROSCEDASTICITY TESTS
# ==============================

print("\n===== HETEROSCEDASTICITY TESTS =====")

# Goldfeld-Quandt
gq = diag.het_goldfeldquandt(Y, X)
print("\nGoldfeld-Quandt Test:")
print(f"F-stat: {gq[0]}, p-value: {gq[1]}")

# White Test
white = diag.het_white(residuals, X)
print("\nWhite Test:")
print(f"LM Stat: {white[0]}, LM p-value: {white[1]}")
print(f"F Stat: {white[2]}, F p-value: {white[3]}")

# ==============================
# 4. AUTOCORRELATION TESTS
# ==============================

print("\n===== AUTOCORRELATION TESTS =====")

# Durbin-Watson
dw = stattools.durbin_watson(residuals)
print(f"\nDurbin-Watson: {dw}")

# Breusch-Godfrey
bg = diag.acorr_breusch_godfrey(model, nlags=2)
print("\nBreusch-Godfrey Test:")
print(f"LM Stat: {bg[0]}, LM p-value: {bg[1]}")
print(f"F Stat: {bg[2]}, F p-value: {bg[3]}")

# ==============================
# 5. NORMALITY TEST
# ==============================

print("\n===== NORMALITY TEST =====")

jb = stattools.jarque_bera(residuals)
print("\nJarque-Bera Test:")
print(f"JB Stat: {jb[0]}, p-value: {jb[1]}")
print(f"Skew: {jb[2]}, Kurtosis: {jb[3]}")

# ==============================
# 6. MULTICOLLINEARITY (VIF)
# ==============================

print("\n===== VIF (MULTICOLLINEARITY) =====")

vif_df = pd.DataFrame()
vif_df["Variable"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i)
                 for i in range(X.shape[1])]
print(vif_df)

# ==============================
# 7. PARAMETER STABILITY (CHOW TEST)
# ==============================

print("\n===== CHOW TEST (STRUCTURAL BREAK) =====")

split_index = int(len(df) * 0.7)

df1 = df.iloc[:split_index]
df2 = df.iloc[split_index:]

Y1 = df1[DEPENDENT_VAR]
X1 = sm.add_constant(df1[INDEPENDENT_VARS])

Y2 = df2[DEPENDENT_VAR]
X2 = sm.add_constant(df2[INDEPENDENT_VARS])

model_full = sm.OLS(Y, X).fit()
model1 = sm.OLS(Y1, X1).fit()
model2 = sm.OLS(Y2, X2).fit()

RSS = sum(model_full.resid**2)
RSS1 = sum(model1.resid**2)
RSS2 = sum(model2.resid**2)

chow_num = (RSS - (RSS1 + RSS2)) / k
chow_den = (RSS1 + RSS2) / (T - 2*k)
chow_stat = chow_num / chow_den

print(f"Chow Statistic: {chow_stat}")

# ==============================
# 8. ROBUST STANDARD ERRORS
# ==============================

print("\n===== ROBUST STANDARD ERRORS (HC3) =====")

robust_model = model.get_robustcov_results(cov_type='HC3')
print(robust_model.summary())

# ==============================
# 9. DIAGNOSTIC PLOTS
# ==============================

print("\n===== GENERATING DIAGNOSTIC PLOTS =====")

# Residuals vs Fitted
plt.figure()
plt.scatter(fitted, residuals)
plt.axhline(0)
plt.title("Residuals vs Fitted")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

# Residuals over time
plt.figure()
plt.plot(residuals)
plt.title("Residuals Over Time")
plt.xlabel("Observation")
plt.ylabel("Residual")
plt.show()

# Histogram of residuals
plt.figure()
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.show()

print("\n===== WORKFLOW COMPLETE =====")
