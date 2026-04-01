import pandas as pd
import numpy as np

# Step 1: Load Excel file
# Make sure your Excel file has a column like 'Close' or 'Price'
file_path = "stock_data.xlsx"
df = pd.read_excel(file_path)

# Step 2: Ensure data is sorted by date (if date column exists)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

# Step 3: Use the price column (change 'Close' if needed)
prices = df['Close']

# Step 4: Calculate log returns
log_returns = np.log(prices / prices.shift(1)).dropna()

# Step 5: Calculate volatility
daily_volatility = log_returns.std()
annual_volatility = daily_volatility * np.sqrt(252)

# Step 6: Rolling volatility (20-day)
rolling_volatility = log_returns.rolling(window=20).std() * np.sqrt(252)

# Step 7: Print results
print("Daily Volatility:", daily_volatility)
print("Annual Volatility:", annual_volatility)

# Step 8: Save results back to Excel (optional)
df['Log_Returns'] = log_returns
df['Rolling_Volatility'] = rolling_volatility

output_file = "volatility_output.xlsx"
df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")