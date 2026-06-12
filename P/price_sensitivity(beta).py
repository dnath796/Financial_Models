import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

# =========================================================
# 1. DATA LOADER (ROBUST)
# =========================================================
def get_data(tickers, start="2002-01-01"):
    raw = yf.download(tickers, start=start, progress=False)

    # Handle MultiIndex safely
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.levels[0]:
            prices = raw["Adj Close"]
        else:
            prices = raw["Close"]
    else:
        prices = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]

    returns = prices.pct_change().dropna()

    # Ensure consistent naming
    returns = returns.copy()
    return returns


# =========================================================
# 2. SAFE BETA FUNCTIONS
# =========================================================
def beta_cov(stock, market):
    return np.cov(stock, market)[0, 1] / np.var(market)


def beta_corr(stock, market):
    corr = np.corrcoef(stock, market)[0, 1]
    return corr * (np.std(stock) / np.std(market))


def beta_downside(stock, market):
    mask = market < 0
    return np.cov(stock[mask], market[mask])[0, 1] / np.var(market[mask])


def beta_ols(stock, market):
    X = sm.add_constant(market)
    model = sm.OLS(stock, X).fit()

    # SAFE extraction (NO KeyError ever)
    return model.params.iloc[1]


# =========================================================
# 3. ROLLING BETA (TIME-VARYING)
# =========================================================
def rolling_beta(stock, market, window=60):
    betas = []

    for i in range(window, len(stock)):
        y = stock.iloc[i-window:i]
        x = market.iloc[i-window:i]

        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        betas.append(model.params.iloc[1])

    return pd.Series(betas, index=stock.index[window:])


# =========================================================
# 4. CAPM MODEL
# =========================================================
def capm(rf, beta, market_return):
    return rf + beta * (market_return - rf)


# =========================================================
# 5. PORTFOLIO BETA
# =========================================================
def portfolio_beta(weights, betas):
    return np.dot(weights, betas)


# =========================================================
# 6. MAIN ENGINE
# =========================================================
if __name__ == "__main__":

    # -------------------------------
    # INPUT
    # -------------------------------
    tickers = ["AAPL", "^GSPC"]
    returns = get_data(tickers)

    stock = returns["AAPL"]
    market = returns["^GSPC"]

    # -------------------------------
    # BETA CALCULATIONS
    # -------------------------------
    b_cov = beta_cov(stock, market)
    b_ols = beta_ols(stock, market)
    b_corr = beta_corr(stock, market)
    b_down = beta_downside(stock, market)

    print("Always consider the context and data quality when choosing a Beta estimation method.")
    print("Covariance-based Beta can be unstable if market variance is low.")
    print("Note: OLS Beta is generally more robust and widely used in practice.")
    print("Correlation-based Beta can be misleading if volatilities differ significantly.")
    print("Downside Beta can be higher if the stock is more sensitive during market downturns.")

    print("\n📊 BETA COMPARISON")
    print("Cov Beta      :", round(b_cov, 4))
    print("OLS Beta      :", round(b_ols, 4))
    print("Corr Beta     :", round(b_corr, 4))
    print("Downside Beta :", round(b_down, 4))

   
    print("Beta: How risky is this stock vs market?", "High" if b_ols > 1 else "Low")

    print("Downside Beta: Is it crash-sensitive?", "Yes" if b_down > b_ols else "No")

    # -------------------------------
    # CAPM
    # -------------------------------
    rf = 0.04
    market_ret = market.mean() * 252

    capm_return = capm(rf, b_ols, market_ret)

    print("\n CAPM Expected Return:", round(capm_return, 4))
    print("CAPM Return: Is this a good investment?", "Yes" if capm_return > rf else "No")
    print("What return should I expect?", "High" if capm_return > 0.1 else "Low")

    # -------------------------------
    # ROLLING BETA
    # -------------------------------
    rb = rolling_beta(stock, market)
    print("\n Latest Rolling Beta:")
    print(rb.tail())
    print("Is this risk stable over time?", "Yes" if rb.std() < 0.1 else "No")

    # -------------------------------
    # PORTFOLIO EXAMPLE
    # -------------------------------
    weights = np.array([0.5, 0.5])
    betas = np.array([1.2, 1.0])

    print("\nPortfolio Beta:", round(portfolio_beta(weights, betas), 4))
    print("How risky is my portfolio?,", "High" if portfolio_beta(weights, betas) > 1 else "Low")