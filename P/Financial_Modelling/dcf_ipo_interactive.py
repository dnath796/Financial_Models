import numpy as np
import pandas as pd

def get_int(prompt):
    return int(input(prompt))

def get_float(prompt):
    return float(input(prompt))

def yes_no(prompt):
    return input(prompt).strip().lower() in ["y", "yes"]

def main():
    print("=== Interactive DCF / IPO Valuation (Revenue + FCF) ===\n")

    # -----------------------------
    # 1) Historical data
    # -----------------------------
    n_years = get_int("How many historical years of data do you have? ")

    hist_years = []
    hist_revenue = []
    hist_fcf = []

    print("\nEnter historical Revenue and FCF (in millions of USD):")
    for i in range(n_years):
        y = get_int(f"  Year {i+1} (e.g., 2020): ")
        rev_y = get_float(f"    Revenue for {y} (in millions): ")
        fcf_y = get_float(f"    FCF for {y} (in millions): ")
        hist_years.append(y)
        hist_revenue.append(rev_y)
        hist_fcf.append(fcf_y)

    hist_df = pd.DataFrame(
        {"year": hist_years, "revenue": hist_revenue, "fcf": hist_fcf}
    ).set_index("year").sort_index()

    # Historical FCF growth and FCF margin
    hist_df["fcf_growth"] = hist_df["fcf"].pct_change()
    avg_fcf_growth = hist_df["fcf_growth"].dropna().mean()
    hist_df["fcf_margin"] = hist_df["fcf"] / hist_df["revenue"]
    avg_margin = hist_df["fcf_margin"].mean()

    print(f"\nAverage historical FCF growth: {avg_fcf_growth:.4f} ({avg_fcf_growth*100:.2f}%)")
    print(f"Average historical FCF margin: {avg_margin:.4f} ({avg_margin*100:.2f}%)")

    # ------------------------------------
    # 2) Forecast mode choice
    # ------------------------------------
    print("\nHow do you want to forecast revenue and FCF?")
    print("  1 = Use a constant revenue growth rate")
    print("  2 = Manually enter revenue for each future year")
    mode = get_int("Choose 1 or 2: ")

    forecast_horizon = get_int("\nHow many future years do you want to forecast? ")

    # ------------------------------------
    # 3) Build revenue & FCF forecast
    # ------------------------------------
    last_year = hist_df.index[-1]
    future_years = []
    future_revenue = []
    future_fcf = []

    if mode == 1:
        # Constant revenue growth + FCF margin
        rev_growth = get_float("\nEnter annual revenue growth rate (e.g. 0.06 for 6%): ")

        # Optionally override FCF margin
        use_custom_margin = yes_no(
            f"Use custom FCF margin instead of average {avg_margin:.4f}? (y/n): "
        )
        if use_custom_margin:
            avg_margin = get_float("  Enter FCF margin (e.g. 0.15 for 15%): ")

        rev_t = hist_df["revenue"].iloc[-1]
        for i in range(1, forecast_horizon + 1):
            year = last_year + i
            rev_t *= (1 + rev_growth)
            fcf_t = rev_t * avg_margin
            future_years.append(year)
            future_revenue.append(rev_t)
            future_fcf.append(fcf_t)

    elif mode == 2:
        # Manual revenue + FCF margin
        use_custom_margin = yes_no(
            f"\nUse custom FCF margin instead of average {avg_margin:.4f}? (y/n): "
        )
        if use_custom_margin:
            avg_margin = get_float("  Enter FCF margin (e.g. 0.15 for 15%): ")

        print("\nEnter future revenue (in millions). FCF will be Revenue * FCF margin.")
        for i in range(1, forecast_horizon + 1):
            year = last_year + i
            rev_y = get_float(f"  Revenue for {year} (in millions): ")
            fcf_y = rev_y * avg_margin
            future_years.append(year)
            future_revenue.append(rev_y)
            future_fcf.append(fcf_y)
    else:
        print("Invalid forecast mode.")
        return

    forecast_df = pd.DataFrame(
        {"year": future_years, "revenue": future_revenue, "fcf": future_fcf}
    ).set_index("year")

    # ------------------------------------
    # 4) Valuation assumptions
    # ------------------------------------
    print("\nValuation assumptions:")
    wacc = get_float("  Discount rate (WACC / cost of equity, e.g. 0.10 for 10%): ")
    terminal_growth = get_float("  Terminal FCF growth rate (e.g. 0.03 for 3%): ")

    print("\nCapital structure:")
    net_debt = get_float("  Net debt (Debt - Cash, in millions): ")

    print("\nShare information:")
    shares_pre_ipo = get_float("  Existing shares BEFORE IPO (units): ")
    primary_shares_ipo = get_float("  New primary shares issued in IPO (units): ")
    total_shares_post_ipo = shares_pre_ipo + primary_shares_ipo

    # ------------------------------------
    # 5) DCF valuation
    # ------------------------------------
    fcfs = forecast_df["fcf"].values  # in millions
    years = np.arange(1, len(fcfs) + 1)
    discount_factors = 1 / (1 + wacc) ** years

    pv_fcfs = (fcfs * discount_factors).sum()

    last_fcf_forecast = fcfs[-1]
    terminal_value = last_fcf_forecast * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** years[-1]

    enterprise_value = pv_fcfs + pv_terminal  # millions
    equity_value = enterprise_value - net_debt  # millions
    equity_value_usd = equity_value * 1_000_000
    fair_price_per_share = equity_value_usd / total_shares_post_ipo

    # ------------------------------------
    # 6) Output
    # ------------------------------------
    print("\n=== Historical Data (millions) ===")
    print(hist_df[["revenue", "fcf"]])

    print("\n=== Forecast Data (millions) ===")
    print(forecast_df[["revenue", "fcf"]])

    print("\n=== Valuation Results (post-IPO) ===")
    print(f"Enterprise value:        ${enterprise_value:,.2f} million")
    print(f"Equity value:            ${equity_value:,.2f} million")
    print(f"Total shares post-IPO:   {total_shares_post_ipo:,.0f}")
    print(f"Implied IPO price/share: ${fair_price_per_share:,.2f}")

if __name__ == "__main__":
    main()
    print("⚠ Retained earnings declining → Possible losses or high payouts.")