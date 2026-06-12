# =========================================
# COMPLETE MULTIPLE LINEAR REGRESSION 
# =========================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.stattools import durbin_watson

# -----------------------------------------
# 1. Load Data
# -----------------------------------------
data = pd.read_csv("data.csv")

# Independent variables
X = data[['TV','Radio','Newspaper']]   # change variables if needed

# Dependent variable
y = data['Sales']

# Add constant (Intercept)
X = sm.add_constant(X)

# -----------------------------------------
# 2. Fit Model
# -----------------------------------------
model = sm.OLS(y, X).fit()

# Full regression output (contains coefficients, p-values, R2, Adj R2, F-test)
print(model.summary())

# -----------------------------------------
# 3. Multicollinearity Test (VIF)
# -----------------------------------------
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i)
              for i in range(X.shape[1])]

print("\nVIF Results")
print(vif)

# -----------------------------------------
# 4. Heteroskedasticity Test (Breusch-Pagan)
# -----------------------------------------
bp_test = het_breuschpagan(model.resid, model.model.exog)
bp_labels = ['LM Statistic','LM-Test p-value','F-Statistic','F-Test p-value']

print("\nBreusch-Pagan Test")
print(dict(zip(bp_labels, bp_test)))

# -----------------------------------------
# 5. Normality Test (Jarque-Bera)
# -----------------------------------------
jb_test = jarque_bera(model.resid)
jb_labels = ['JB Statistic','JB p-value','Skew','Kurtosis']

print("\nJarque-Bera Test")
print(dict(zip(jb_labels, jb_test)))

# -----------------------------------------
# 6. Autocorrelation Tests
# -----------------------------------------
print("\nDurbin-Watson:", durbin_watson(model.resid))

bg_test = acorr_breusch_godfrey(model, nlags=2)
bg_labels = ['LM Statistic','LM-Test p-value','F-Statistic','F-Test p-value']

print("\nBreusch-Godfrey Test")
print(dict(zip(bg_labels, bg_test)))

# -----------------------------------------
# 7. Prediction Example
# -----------------------------------------
new_data = pd.DataFrame({
    'const':[1],
    'TV':[200],
    'Radio':[40],
    'Newspaper':[30]
})

prediction = model.predict(new_data)
print("\nPrediction:", prediction)
