

# Margin Trading and Short Sale Calculations

# -------- Margin Trade --------
initial_margin_pct = 0.65
maintenance_margin_pct = 0.35
shares_purchased = 300
initial_price = 50
current_price = 75
loan_rate = 0.12
holding_period = 1

initial_stock_value = shares_purchased * initial_price
initial_equity = initial_margin_pct * initial_stock_value
amount_borrowed = initial_stock_value - initial_equity

new_stock_value = shares_purchased * current_price
equity_after = new_stock_value - amount_borrowed
margin_pct_after = equity_after / new_stock_value

interest = amount_borrowed * loan_rate * holding_period
net_profit_margin = equity_after - initial_equity - interest
return_with_margin = net_profit_margin / initial_equity

profit_without_margin = new_stock_value - initial_stock_value
return_without_margin = profit_without_margin / initial_stock_value

print("----- Margin Trade -----")
print(f"Initial Equity: ${initial_equity:,.2f}")
print(f"Amount Borrowed: ${amount_borrowed:,.2f}")
print(f"Equity After Price Change: ${equity_after:,.2f}")
print(f"Margin % After Price Change: {margin_pct_after*100:.2f}%")
print(f"Return With Margin: {return_with_margin*100:.2f}%")
print(f"Return Without Margin: {return_without_margin*100:.2f}%")


# -------- Short Sale --------
initial_margin_pct_short = 0.50
maintenance_margin_pct_short = 0.30
shares_shorted = 100
initial_short_price = 100
current_short_price = 130

sale_proceeds = shares_shorted * initial_short_price
initial_margin_deposit = initial_margin_pct_short * sale_proceeds
total_account_value = sale_proceeds + initial_margin_deposit

stock_owed_after = shares_shorted * current_short_price
net_equity_short = total_account_value - stock_owed_after
margin_pct_short = net_equity_short / stock_owed_after

loss_short = initial_margin_deposit - net_equity_short
return_short = -loss_short / initial_margin_deposit

print("\n----- Short Sale -----")
print(f"Sale Proceeds: ${sale_proceeds:,.2f}")
print(f"Initial Margin Deposit: ${initial_margin_deposit:,.2f}")
print(f"Stock Owed After Price Change: ${stock_owed_after:,.2f}")
print(f"Net Equity After Price Change: ${net_equity_short:,.2f}")
print(f"Margin % After Price Change: {margin_pct_short*100:.2f}%")
print(f"Return on Short Sale: {return_short*100:.2f}%")
