import numpy as np
import pandas as pd

# ========= CONFIG =========
XLSX_PATH = "fcf_history.xlsx"   # path to your Excel file
SHEET_NAME = 0                   # 0 = first sheet; or use "Sheet1"


def get_float(prompt):
    return float(input(prompt))


def yes_no(prompt):
    return input(prompt).strip().lower() in ["y", "yes"]


def main():
    print("=== FCF Growth from XLSX + DCF Valuation ===\n")

    # ---------------------------------
    # 1) Load historical FCF from XLSX
    # ---------------------------------
    # Expected columns: year, fcf  (FCF in millions)
    # If you get engine errors, ensure `openpyxl` is installed, and add engine="openpyxl"
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)  # , engine="openpyxl"

    df.columns = df.columns.str.strip() 

    required_cols = {"year", "fcf"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Excel sheet must contain columns: {required_cols}, found: {df.columns.tolist()}"
        )

    df = df.sort_values("year").set_index("year")
    print("Loaded historical FCF (millions):")
    print(df)

    # ---------------------------------
    # 2) Compute historical FCF growth
    # ---------------------------------
    df["fcf_growth"] = df["fcf"].pct_change()
    growth_series = df["fcf_growth"].dropna()

    if growth_series.empty:
        raise ValueError("Need at least 2 years of FCF data to compute growth rates.")

    g_avg = growth_series.mean()

    lower = growth_series.quantile(0.10)
    upper = growth_series.quantile(0.90)
    trimmed = growth_series[(growth_series >= lower) & (growth_series <= upper)]
    g_trimmed = trimmed.mean()

    print("\nHistorical FCF growth rates:")
    print(growth_series)
    print(f"\nSimple average growth:  {g_avg:.4f} ({g_avg*100:.2f}%)")
    print(f"Trimmed average growth: {g_trimmed:.4f} ({g_trimmed*100:.2f}%)")

    # ---------------------------------
    # 3) Choose forecast growth rate
    # ---------------------------------
    print("\nChoose FCF growth rate for forecasting:")
    print("  1 = Simple average historical growth")
    print("  2 = Trimmed average historical growth (ignores extremes)")
    print("  3 = Custom growth rate (manual)")
    mode = int(input("Select 1, 2, or 3: "))

    if mode == 1:
        g_forecast = g_avg
    elif mode == 2:
        g_forecast = g_trimmed
    elif mode == 3:
        g_forecast = get_float("Enter annual FCF growth rate (e.g. 0.05 for 5%): ")
    else:
        raise ValueError("Invalid selection for growth rate.")

    print(f"\nUsing forecast FCF growth rate: {g_forecast:.4f} ({g_forecast*100:.2f}%)")

    forecast_horizon = int(input("\nHow many future years do you want to forecast? "))

    # ---------------------------------
    # 4) Build FCF forecast from trend
    # ---------------------------------
    last_year = df.index[-1]
    last_fcf = df["fcf"].iloc[-1]

    future_years = []
    future_fcfs = []

    fcf_t = last_fcf
    for i in range(1, forecast_horizon + 1):
        year = last_year + i
        fcf_t *= (1 + g_forecast)
        future_years.append(year)
        future_fcfs.append(fcf_t)

    forecast_df = pd.DataFrame(
        {"year": future_years, "fcf": future_fcfs}
    ).set_index("year")

    print("\nForecast FCF (millions):")
    print(forecast_df)

    # ---------------------------------
    # 5) Valuation assumptions
    # ---------------------------------
    print("\nValuation assumptions:")
    wacc = get_float("  Discount rate (e.g. 0.10 for 10%): ")
    terminal_growth = get_float("  Terminal FCF growth rate (e.g. 0.03 for 3%): ")

    print("\nCapital structure:")
    net_debt = get_float("  Net debt (Debt - Cash, in millions): ")

    print("\nShare information:")
    shares_outstanding = get_float("  Total shares outstanding (units): ")

    # ---------------------------------
    # 6) DCF valuation
    # ---------------------------------
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
    fair_price_per_share = equity_value_usd / shares_outstanding

    # ---------------------------------
    # 7) Output
    # ---------------------------------
    print("\n=== Historical Data (millions) ===")
    print(df[["fcf", "fcf_growth"]])

    print("\n=== Forecast FCF (millions) ===")
    print(forecast_df)

    print("\n=== Valuation Results ===")
    print(f"Enterprise value:      ${enterprise_value:,.2f} million")
    print(f"Equity value:          ${equity_value:,.2f} million")
    print(f"Fair value per share:  ${fair_price_per_share:,.2f}")


if __name__ == "__main__":
    main()
