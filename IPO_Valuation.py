import numpy as np
import pandas as pd

# ---------- Inputs (customize for your IPO) ----------

# Projected free cash flows to equity (or firm) in USD millions
fcf_forecast = [10, 14, 18, 22, 26]   # years 1–5

discount_rate = 0.12                  # required return (ke, or WACC if valuing firm)
terminal_growth = 0.03                # long‑run g for Gordon growth
net_debt = 50                         # debt – cash, in millions (0 if using FCFE)
shares_outstanding_post_ipo = 80_000_000
ipo_primary_shares = 20_000_000       # new shares issued in IPO

# Trading comparables (US peers, e.g., same sector/size)
peer_pe_multiples = [18, 20, 22]      # forward P/E
peer_ev_ebitda_multiples = [9, 10, 11]

# Company’s next‑12‑month earnings and EBITDA (in millions)
forward_net_income = 20
forward_ebitda = 35

# Weights for combined valuation
weight_dcf = 0.5
weight_pe = 0.25
weight_ev_ebitda = 0.25

# ---------- Valuation functions ----------

def dcf_valuation(fcf_list, r, g, net_debt=0.0, equity_valuation=True):
    """
    Simple DCF with Gordon growth terminal value.
    fcf_list in millions, r and g as decimals.
    If equity_valuation is False, returns enterprise value.
    """
    years = np.arange(1, len(fcf_list) + 1)
    fcf = np.array(fcf_list, dtype=float)

    # Present value of explicit-period cash flows
    discount_factors = 1 / (1 + r) ** years
    pv_fcf = np.sum(fcf * discount_factors)

    # Terminal value at end of last forecast year
    tv = fcf[-1] * (1 + g) / (r - g)
    pv_tv = tv / (1 + r) ** years[-1]

    enterprise_value = pv_fcf + pv_tv

    if equity_valuation:
        equity_value = enterprise_value - net_debt
        return equity_value
    else:
        return enterprise_value

def multiple_from_pe(peers_pe, fwd_eps):
    """Equity value using P/E multiple."""
    avg_pe = np.mean(peers_pe)
    return avg_pe * fwd_eps

def multiple_from_ev_ebitda(peers_ev_ebitda, fwd_ebitda, net_debt):
    """Equity value from EV/EBITDA multiple."""
    avg_ev_ebitda = np.mean(peers_ev_ebitda)
    ev = avg_ev_ebitda * fwd_ebitda
    equity_value = ev - net_debt
    return equity_value

# ---------- Run valuation ----------

# 1) DCF (treating FCF as to firm, then subtract net debt)
dcf_equity_value_mn = dcf_valuation(
    fcf_list=fcf_forecast,
    r=discount_rate,
    g=terminal_growth,
    net_debt=net_debt,
    equity_valuation=True,
)

# 2) P/E multiple valuation
total_shares_post_ipo = shares_outstanding_post_ipo + ipo_primary_shares
fwd_eps = forward_net_income / (total_shares_post_ipo / 1_000_000)  # EPS in USD, shares in millions
pe_equity_value_mn = multiple_from_pe(peer_pe_multiples, fwd_eps)

# 3) EV/EBITDA multiple valuation
ev_ebitda_equity_value_mn = multiple_from_ev_ebitda(
    peer_ev_ebitda_multiples,
    forward_ebitda,
    net_debt
)

# 4) Combine methods
weighted_equity_value_mn = (
    weight_dcf * dcf_equity_value_mn
    + weight_pe * pe_equity_value_mn
    + weight_ev_ebitda * ev_ebitda_equity_value_mn
)

# 5) Implied IPO price per share
implied_ipo_price = (weighted_equity_value_mn * 1_000_000) / total_shares_post_ipo

print(f"DCF equity value: ${dcf_equity_value_mn:,.1f} million")
print(f"P/E equity value: ${pe_equity_value_mn:,.1f} million")
print(f"EV/EBITDA equity value: ${ev_ebitda_equity_value_mn:,.1f} million")
print(f"Weighted equity value: ${weighted_equity_value_mn:,.1f} million")
print(f"Implied IPO offer price: ${implied_ipo_price:,.2f} per share")

