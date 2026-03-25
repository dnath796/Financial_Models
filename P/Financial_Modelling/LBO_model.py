import numpy as np
import pandas as pd
import numpy_financial as npf

# -----------------------------
# 1. Assumptions
# -----------------------------
years = 5

# Operating
revenue = 500
revenue_growth = 0.04
ebitda_margin = 0.20
tax_rate = 0.25

capex_pct = 0.03
wc_pct = 0.02   # % of revenue increase

# Entry / Exit
entry_multiple = 10
exit_multiple = 10

# Capital structure
debt_pct = 0.7
interest_rate = 0.08

# -----------------------------
# 2. Entry Valuation
# -----------------------------
ebitda_0 = revenue * ebitda_margin
entry_ev = ebitda_0 * entry_multiple

debt = entry_ev * debt_pct
equity = entry_ev - debt

# -----------------------------
# 3. Projection Setup
# -----------------------------
data = []
beginning_debt = debt
prev_revenue = revenue

for t in range(1, years + 1):
    # Revenue growth
    revenue = prev_revenue * (1 + revenue_growth)
    
    # EBITDA
    ebitda = revenue * ebitda_margin
    
    # Depreciation (simplified = capex)
    capex = revenue * capex_pct
    depreciation = capex
    
    # EBIT
    ebit = ebitda - depreciation
    
    # Interest
    interest = beginning_debt * interest_rate
    
    # Taxes
    ebt = ebit - interest
    taxes = max(0, ebt * tax_rate)
    
    # Net Income
    net_income = ebt - taxes
    
    # Change in Working Capital
    delta_wc = (revenue - prev_revenue) * wc_pct
    
    # Free Cash Flow (FCF)
    fcf = net_income + depreciation - capex - delta_wc
    
    # Cash sweep to repay debt
    repayment = max(0, min(fcf, beginning_debt))
    ending_debt = beginning_debt - repayment
    
    data.append({
        "Year": t,
        "Revenue": revenue,
        "EBITDA": ebitda,
        "EBIT": ebit,
        "Interest": interest,
        "Taxes": taxes,
        "FCF": fcf,
        "Debt Start": beginning_debt,
        "Repayment": repayment,
        "Debt End": ending_debt
    })
    
    prev_revenue = revenue
    beginning_debt = ending_debt

df = pd.DataFrame(data)

# -----------------------------
# 4. Exit
# -----------------------------
exit_ebitda = df.iloc[-1]["EBITDA"]
exit_ev = exit_ebitda * exit_multiple
exit_debt = df.iloc[-1]["Debt End"]

exit_equity = exit_ev - exit_debt

# -----------------------------
# 5. IRR & MOIC
# -----------------------------
cash_flows = [-equity] + [0]*years
cash_flows[-1] = exit_equity

irr = npf.irr(cash_flows)
moic = exit_equity / equity

# -----------------------------
# 6. Results
# -----------------------------
print("===== LBO SUMMARY =====")
print(f"Entry EV: {entry_ev:.2f}")
print(f"Equity Invested: {equity:.2f}")
print(f"Exit EV: {exit_ev:.2f}")
print(f"Exit Equity: {exit_equity:.2f}")
print(f"MOIC: {moic:.2f}x")
print(f"IRR: {irr*100:.2f}%\n")

print("===== PROJECTIONS =====")
print(df)