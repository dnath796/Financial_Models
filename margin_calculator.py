# ================== MARGIN TRADING ==================
print("===== MARGIN TRADING CALCULATOR =====")

initial_margin_pct = float(input("Initial Margin % (e.g., 65): ")) / 100
maintenance_margin_pct = float(input("Maintenance Margin % (e.g., 35): ")) / 100
shares = int(input("Number of Shares Purchased: "))
initial_price = float(input("Initial Stock Price: "))
current_price = float(input("Current Stock Price: "))
loan_rate = float(input("Loan Interest Rate % (Simple): ")) / 100
time = float(input("Holding Period (years): "))

initial_value = shares * initial_price
equity_initial = initial_margin_pct * initial_value
loan = initial_value - equity_initial

new_value = shares * current_price
equity_now = new_value - loan
margin_pct_now = equity_now / new_value

interest = loan * loan_rate * time
profit_margin = equity_now - equity_initial - interest
return_margin = profit_margin / equity_initial

profit_cash = new_value - initial_value
return_cash = profit_cash / initial_value

margin_call = 1 if margin_pct_now < maintenance_margin_pct else 0

# 🔹 Margin Call Trigger Price (LONG)
trigger_price_long = loan / (shares * (1 - maintenance_margin_pct))

print("\n----- Margin Trade Results -----")
print(f"Initial Equity: ${equity_initial:,.2f}")
print(f"Loan Amount: ${loan:,.2f}")
print(f"Equity After Price Change: ${equity_now:,.2f}")
print(f"Margin % Now: {margin_pct_now*100:.2f}%")
print(f"Margin Call (0=No, 1=Yes): {margin_call}")
print(f"Return With Margin: {return_margin*100:.2f}%")
print(f"Return Without Margin: {return_cash*100:.2f}%")
print(f"⚠️ Margin Call Trigger Price (Stock Falls To): ${trigger_price_long:,.2f}")


# ================== SHORT SALE ==================
print("\n===== SHORT SALE CALCULATOR =====")

init_margin_short = float(input("Initial Margin % (e.g., 50): ")) / 100
maint_margin_short = float(input("Maintenance Margin % (e.g., 30): ")) / 100
shares_short = int(input("Number of Shares Shorted: "))
short_price_initial = float(input("Initial Short Sale Price: "))
short_price_current = float(input("Current Stock Price: "))

sale_proceeds = shares_short * short_price_initial
margin_deposit = init_margin_short * sale_proceeds
account_total = sale_proceeds + margin_deposit

stock_owed_now = shares_short * short_price_current
equity_short_now = account_total - stock_owed_now
margin_pct_short = equity_short_now / stock_owed_now

margin_call_short = 1 if margin_pct_short < maint_margin_short else 0

profit_short = equity_short_now - margin_deposit
return_short = profit_short / margin_deposit

# 🔹 Margin Call Trigger Price (SHORT)
trigger_price_short = account_total / (shares_short * (1 + maint_margin_short))

print("\n----- Short Sale Results -----")
print(f"Sale Proceeds: ${sale_proceeds:,.2f}")
print(f"Margin Deposit: ${margin_deposit:,.2f}")
print(f"Stock Owed Now: ${stock_owed_now:,.2f}")
print(f"Net Equity: ${equity_short_now:,.2f}")
print(f"Margin % Now: {margin_pct_short*100:.2f}%")
print(f"Margin Call (0=No, 1=Yes): {margin_call_short}")
print(f"Return on Short Sale: {return_short*100:.2f}%")
print(f"⚠️ Margin Call Trigger Price (Stock Rises To): ${trigger_price_short:,.2f}")


# ================== SCENARIO ANALYSIS ==================
print("\n===== PROFIT / LOSS SCENARIO ANALYSIS =====")
test_price = float(input("Enter a hypothetical future stock price to test P/L: "))

# Long position P/L
test_value_long = shares * test_price
test_equity_long = test_value_long - loan
test_profit_long = test_equity_long - equity_initial - interest

# Short position P/L
test_value_short = shares_short * test_price
test_equity_short = account_total - test_value_short
test_profit_short = test_equity_short - margin_deposit

print("\nIf stock price becomes ${:,.2f}:".format(test_price))
print(f"Long Margin Trade Profit/Loss: ${test_profit_long:,.2f}")
print(f"Short Sale Profit/Loss: ${test_profit_short:,.2f}")
