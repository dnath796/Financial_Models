import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. LOAD DATA
# -----------------------------
file_path = "P/P/Sample_Data.csv"
df = pd.read_csv(file_path)

# Sort by year (important for growth calculations)
df = df.sort_values("Year")

print("\n📄 Original Data:")
print(df)


# -----------------------------
# 2. GROWTH CALCULATIONS
# -----------------------------
df["Asset_Growth_%"] = df["Total_Assets"].pct_change() * 100
df["Equity_Growth_%"] = df["Shareholders_Equity"].pct_change() * 100
df["Retained_Earnings_Growth_%"] = df["Retained_Earnings"].pct_change() * 100


# -----------------------------
# 3. FINANCIAL STRENGTH RATIOS
# -----------------------------
df["Debt_to_Equity"] = df["Total_Liabilities"] / df["Shareholders_Equity"]
df["Debt_Ratio"] = df["Total_Liabilities"] / df["Total_Assets"]
df["Working_Capital"] = df["Cash"] + df["Inventory"] - df["Total_Liabilities"] * 0.2  # rough liquidity view


# -----------------------------
# 4. PROFITABILITY RATIOS (IF NET INCOME EXISTS)
# -----------------------------
if "Net_Income" in df.columns:
    df["ROA_%"] = (df["Net_Income"] / df["Total_Assets"]) * 100
    df["ROE_%"] = (df["Net_Income"] / df["Shareholders_Equity"]) * 100


# -----------------------------
# 5. DISPLAY ANALYSIS TABLE
# -----------------------------
print("\n📊 Financial Analysis:")
print(df.round(2))


# -----------------------------
# 6. INTERPRET PROFIT / LOSS TREND
# -----------------------------
print("\n🔍 Profitability Insight:")

if df["Shareholders_Equity"].iloc[-1] > df["Shareholders_Equity"].iloc[0]:
    print("✔ Shareholder equity increased over time → Business likely profitable.")
else:
    print("⚠ Equity decreased → Possible losses or high withdrawals.")

if df["Retained_Earnings"].iloc[-1] > df["Retained_Earnings"].iloc[0]:
    print("✔ Retained earnings growing → Profits are being reinvested.")
else:
    print("⚠ Retained earnings declining → Company may be distributing or losing profits.")

if "Net_Income" in df.columns:
    if df["Net_Income"].mean() > 0:
        print("✔ Net income positive on average → Company is profitable.")
    else:
        print("⚠ Net income negative → Company is operating at a loss.")

# -----------------------------
# 7. PLOTTING TRENDS
# -----------------------------
plt.figure()
plt.plot(df["Year"], df["Shareholders_Equity"], marker='o')
plt.title("Shareholders Equity Growth")
plt.xlabel("Year")
plt.ylabel("Equity")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["Year"], df["Retained_Earnings"], marker='o')
plt.title("Retained Earnings Growth")
plt.xlabel("Year")
plt.ylabel("Retained Earnings")
plt.grid(True)
plt.show()

if "Net_Income" in df.columns:
    plt.figure()
    plt.plot(df["Year"], df["Net_Income"], marker='o')
    plt.title("Net Income Trend")
    plt.xlabel("Year")
    plt.ylabel("Net Income")
    plt.grid(True)
    plt.show()

if "ROE_%" in df.columns:
    plt.figure()
    plt.plot(df["Year"], df["ROE_%"], marker='o')
    plt.title("Return on Equity (ROE %)")
    plt.xlabel("Year")
    plt.ylabel("ROE %")
    plt.grid(True)
    plt.show()
