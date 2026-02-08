# =========================
# Multiple Linear Regression Complete Template
# =========================

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# -------------------------
# 1. Load dataset
# -------------------------
data = pd.read_csv('data.csv')

# Independent variables
X = data[['TV', 'Radio', 'Newspaper']]

# Dependent variable
y = data['Sales']

# Add constant (intercept)
X = sm.add_constant(X)

# -------------------------
# 2. Fit regression model
# -------------------------
model = sm.OLS(y, X).fit()

# Regression summary
print(model.summary())

# -------------------------
# 3. Multicollinearity Check (VIF)
# -------------------------
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]

print("\nVIF Results:")
print(vif_data)

# -------------------------
# 4. Heteroskedasticity Test
# -------------------------
bp_test = het_breuschpagan(model.resid, model.model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

print("\nBreusch-Pagan Test:")
print(dict(zip(labels, bp_test)))

# -------------------------
# 5. Prediction Example
# -------------------------
new_data = pd.DataFrame({
    'const': [1],
    'TV': [200],
    'Radio': [40],
    'Newspaper': [30]
})

prediction = model.predict(new_data)
print("\nPrediction:", prediction)
