import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# 1 Load Data
# -----------------------------

data = pd.read_csv("timeseries_data.csv")

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

series = data['Value']

# -----------------------------
# 2 Plot Time Series
# -----------------------------

plt.figure(figsize=(10,5))
plt.plot(series)
plt.title("Original Time Series")
plt.show()

# -----------------------------
# 3 Stationarity Test (ADF)
# -----------------------------

result = adfuller(series)

print("ADF Statistic:", result[0])
print("p-value:", result[1])

# -----------------------------
# 4 Differencing
# -----------------------------

series_diff = series.diff().dropna()

plt.figure(figsize=(10,5))
plt.plot(series_diff)
plt.title("Differenced Series")
plt.show()

# -----------------------------
# 5 ACF and PACF
# -----------------------------

plot_acf(series_diff, lags=20)
plot_pacf(series_diff, lags=20)
plt.show()

# -----------------------------
# 6 Split Data (Train/Test)
# -----------------------------

train_size = int(len(series) * 0.8)
train = series[:train_size]
test = series[train_size:]

# -----------------------------
# 7 Build ARIMA Model
# -----------------------------

model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())

# -----------------------------
# 8 Residual Analysis
# -----------------------------

residuals = model_fit.resid

plt.figure(figsize=(10,5))
plt.plot(residuals)
plt.title("Residuals")
plt.show()

# -----------------------------
# 9 Ljung-Box Test
# -----------------------------

ljung_box = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("Ljung-Box Test:")
print(ljung_box)

# -----------------------------
# 10 Forecast
# -----------------------------

forecast = model_fit.forecast(steps=len(test))

plt.figure(figsize=(10,5))
plt.plot(train, label="Train")
plt.plot(test, label="Actual")
plt.plot(test.index, forecast, label="Forecast")
plt.legend()
plt.title("ARIMA Forecast")
plt.show()

# -----------------------------
# 11 Exponential Smoothing
# -----------------------------

exp_model = ExponentialSmoothing(train, trend='add')
exp_fit = exp_model.fit()

exp_forecast = exp_fit.forecast(len(test))

# -----------------------------
# 12 Forecast Accuracy
# -----------------------------

mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, forecast)

print("ARIMA Forecast Accuracy")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)

# -----------------------------
# 13 Sign Prediction Accuracy
# -----------------------------

actual_direction = np.sign(test.diff().dropna())
forecast_direction = np.sign(pd.Series(forecast).diff().dropna())

correct = (actual_direction == forecast_direction).sum()

accuracy = correct / len(actual_direction)

print("Sign Prediction Accuracy:", accuracy)

# -----------------------------
# 14 Compare with Exponential Smoothing
# -----------------------------

mse_exp = mean_squared_error(test, exp_forecast)
print("Exponential Smoothing MSE:", mse_exp)
