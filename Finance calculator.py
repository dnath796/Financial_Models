import matplotlib.pyplot as plt
import csv
import numpy as np
import os

import csv

def plot_short_graph():
    prices = []
    profits = []

    with open("short_position_analysis.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            prices.append(float(row[0]))
            profits.append(float(row[1]))

    plt.figure()
    plt.plot(prices, profits)
    plt.axhline(0)
    plt.title("Short Sale Position: Profit/Loss vs Stock Price")
    plt.xlabel("Stock Price")
    plt.ylabel("Profit / Loss ($)")
    plt.show()

def plot_long_graph():
    prices = []
    profits = []

    with open("long_position_analysis.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            prices.append(float(row[0]))
            profits.append(float(row[1]))

    plt.figure()
    plt.plot(prices, profits)
    plt.axhline(0)
    plt.title("Long Margin Position: Profit/Loss vs Stock Price")
    plt.xlabel("Stock Price")
    plt.ylabel("Profit / Loss ($)")
    plt.show()


def margin_trading():
    print("\n--- Margin Trading Selected ---")

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
    interest = loan * loan_rate * time

    new_value = shares * current_price
    equity_now = new_value - loan
    margin_pct_now = equity_now / new_value
    margin_call = margin_pct_now < maintenance_margin_pct

    profit = equity_now - equity_initial - interest
    ret = profit / equity_initial

    trigger_price = loan / (shares * (1 - maintenance_margin_pct))

    print("\n----- Results -----")
    print(f"Profit/Loss: ${profit:,.2f}")
    print(f"Return: {ret*100:.2f}%")
    print(f"⚠️ Margin Call Trigger Price: ${trigger_price:,.2f}")

    if margin_call:
        print("🚨 WARNING: You are below maintenance margin!")

    scenario_analysis_long(shares, loan, equity_initial, interest, maintenance_margin_pct
)


def short_sale():
    print("\n--- Short Sale Selected ---")

    init_margin = float(input("Initial Margin % (e.g., 50): ")) / 100
    maint_margin = float(input("Maintenance Margin % (e.g., 30): ")) / 100
    shares = int(input("Number of Shares Shorted: "))
    initial_price = float(input("Initial Short Sale Price: "))
    current_price = float(input("Current Stock Price: "))

    sale_proceeds = shares * initial_price
    deposit = init_margin * sale_proceeds
    account_total = sale_proceeds + deposit

    stock_owed = shares * current_price
    equity_now = account_total - stock_owed
    margin_pct = equity_now / stock_owed
    margin_call = margin_pct < maint_margin

    profit = equity_now - deposit
    ret = profit / deposit

    trigger_price = account_total / (shares * (1 + maint_margin))

    print("\n----- Results -----")
    print(f"Profit/Loss: ${profit:,.2f}")
    print(f"Return: {ret*100:.2f}%")
    print(f"⚠️ Margin Call Trigger Price: ${trigger_price:,.2f}")

    if margin_call:
        print("🚨 WARNING: Margin call risk! Add funds immediately!")
    scenario_analysis_short(shares, account_total, deposit
)


def scenario_analysis_long(shares, loan, equity_initial, interest, maint_margin):
    print("\n📊 Profit/Loss Table (Long Position)")
    prices = [p for p in range(10, 151, 10)]

    with open("long_position_analysis.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Price", "Profit/Loss"])

        for price in prices:
            equity = shares * price - loan
            profit = equity - equity_initial - interest
            writer.writerow([price, profit])
            print(f"Price ${price:>3} → P/L = ${profit:,.2f}")

    print("💾 Results saved to long_position_analysis.csv")


def scenario_analysis_short(shares, account_total, deposit):
    print("\n📊 Profit/Loss Table (Short Position)")
    prices = [p for p in range(10, 151, 10)]

    with open("short_position_analysis.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Price", "Profit/Loss"])

        for price in prices:
            equity = account_total - shares * price
            profit = equity - deposit
            writer.writerow([price, profit])
            print(f"Price ${price:>3} → P/L = ${profit:,.2f}")

    print("💾 Results saved to short_position_analysis.csv")


# ================== MAIN MENU LOOP ==================
while True:
    print("\n===== TRADING STRATEGY CALCULATOR =====")
    print("1️⃣1 Margin Trading")
    print("2️⃣ Short Sale")
    print("3️⃣ Exit")

    choice = input("Choose option: ")

    if choice == "1":
        margin_trading()
    elif choice == "2":
        short_sale()
    elif choice == "3":
        print("👋 Exiting program. Stay smart with leverage!")
        break
    else:
        print("❌ Invalid choice. Try again.")


