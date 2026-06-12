import numpy as np

# ---------------------------
# 1) INPUTS (from statements)
# ---------------------------
# Example: last year actuals from income statement (in USD millions)
revenue = 500
operating_income = 80          # EBIT
depreciation_amortization = 20
interest_expense = 5
income_tax_expense = 12
income_before_tax = 60
net_income = 48

# From cash flow & balance sheet (or derived)
change_in_working_capital = 5  # increase in WC (use + if cash outflow)
capex = 25                     # capital expenditures (cash outflow, positive number here)

# Market/valuation inputs
wacc = 0.10                    # discount rate
terminal_growth = 0.03         # long-run FCF growth
forecast_horizon = 5           # years
shares_outstanding = 50_000_000
net_debt = 150                 # debt - cash, in millions


# ----------------------------------
# 2) HELPER: CALCULATE ONE-YEAR FCF
# ----------------------------------
def compute_fcf(net_income,
                depreciation_amortization,
                change_in_working_capital,
                capex):
    """
    Basic Free Cash Flow to Firm (FCFF) approximation from income statement data.
    FCF = Net income + D&A - change in working capital - CapEx
    All inputs in millions.
    """
    fcf = (net_income
           + depreciation_amortization
           - change_in_working_capital
           - capex)
    return fcf


# Compute last-year FCF level
fcf_0 = compute_fcf(
    net_income,
    depreciation_amortization,
    change_in_working_capital,
    capex
)


# ---------------------------------------
# 3) FORECAST FCFs AND DISCOUNT (DCF)
# ---------------------------------------
def project_fcfs(fcf_start, growth_rate, years):
    """Project FCFs assuming constant growth each year."""
    fcfs = []
    fcf_t = fcf_start
    for t in range(1, years + 1):
        fcf_t *= (1 + growth_rate)
        fcfs.append(fcf_t)
    return np.array(fcfs)


def compute_dcf_from_fcfs(fcfs, wacc, terminal_growth):
    """
    Discount projected FCFs and add a terminal value using the
    perpetuity growth model.
    """
    years = np.arange(1, len(fcfs) + 1)
    discount_factors = 1 / (1 + wacc) ** years

    present_value_fcfs = (fcfs * discount_factors).sum()

    # Terminal value at end of last forecast year
    last_fcf = fcfs[-1]
    terminal_value = last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    present_value_tv = terminal_value / (1 + wacc) ** years[-1]

    enterprise_value = present_value_fcfs + present_value_tv
    return enterprise_value


# Example: assume FCF grows at same long-run rate as revenue proxy
fcf_growth_rate = 0.05

fcf_forecast = project_fcfs(fcf_0, fcf_growth_rate, forecast_horizon)
enterprise_value = compute_dcf_from_fcfs(fcf_forecast, wacc, terminal_growth)

# ---------------------------------------
# 4) GO FROM ENTERPRISE VALUE TO PRICE
# ---------------------------------------
equity_value = enterprise_value - net_debt    # in millions
equity_value_usd = equity_value * 1_000_000   # convert to dollars
fair_price_per_share = equity_value_usd / shares_outstanding

print(f"Last-year FCF: ${fcf_0:,.1f} million")
print("Projected FCFs (millions):", np.round(fcf_forecast, 1))
print(f"Enterprise value: ${enterprise_value:,.1f} million")
print(f"Equity value: ${equity_value:,.1f} million")
print(f"Fair value per share: ${fair_price_per_share:,.2f}")
