# forecasting_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

class ForecastingModel:
    def __init__(self, data_path, date_col=None, value_col=None):
        self.data_path = data_path
        self.date_col = date_col
        self.value_col = value_col
        self.ts = self.load_data()
        self.model = None
        self.forecast_values = None

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"CSV file not found at {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Parse date column if provided
        if self.date_col:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df.set_index(self.date_col, inplace=True)
        
        # Select numeric column for analysis
        if self.value_col:
            ts = df[self.value_col]
        else:
            # Default to second column if value_col not provided
            ts = df.select_dtypes(include=[np.number]).iloc[:, 0]
        
        if not np.issubdtype(ts.dtype, np.number):
            raise ValueError("Selected series is not numeric. Please check your CSV or value_col.")
        
        print("Data loaded. First 5 rows:\n", ts.head())
        return ts

    def plot_series(self, title='Time Series'):
        plt.figure(figsize=(10,5))
        plt.plot(self.ts, label='Original Series')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

    def test_stationarity(self):
        print("Performing Augmented Dickey-Fuller test...")
        result = adfuller(self.ts)
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        if result[1] > 0.05:
            print("Series is non-stationary. Differencing is required.")
            return False
        print("Series is stationary.")
        return True

    def difference_series(self):
        self.ts = self.ts.diff().dropna()
        print("Series differenced to achieve stationarity.")
        self.plot_series(title='Differenced Series')

    def plot_acf_pacf(self, lags=20):
        fig, ax = plt.subplots(1,2, figsize=(15,5))
        plot_acf(self.ts, ax=ax[0], lags=lags)
        plot_pacf(self.ts, ax=ax[1], lags=lags)
        plt.show()

    def fit_arima(self, order=(1,1,1)):
        print(f"Fitting ARIMA model with order {order}...")
        self.model = ARIMA(self.ts, order=order).fit()
        print(self.model.summary())

    def residual_diagnostics(self):
        residuals = self.model.resid
        plt.figure(figsize=(10,5))
        plt.plot(residuals)
        plt.title('Residuals')
        plt.show()

        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        print("\nLjung-Box test:\n", lb_test)

    def forecast(self, steps=10):
        self.forecast_values = self.model.forecast(steps=steps)
        print(f"\nForecasted values for next {steps} periods:\n", self.forecast_values)
        self.plot_forecast(steps)
        return self.forecast_values

    def plot_forecast(self, steps=10):
        plt.figure(figsize=(10,5))
        plt.plot(self.ts, label='Historical')
        forecast_index = pd.date_range(self.ts.index[-1], periods=steps+1, freq='D')[1:]
        plt.plot(forecast_index, self.forecast_values, label='Forecast', color='red')
        plt.title('Time Series Forecast')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

    def exponential_smoothing(self, seasonal_periods=None, trend=None, seasonal=None, steps=10):
        print("Fitting Exponential Smoothing model...")
        model_es = ExponentialSmoothing(self.ts, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods).fit()
        forecast_es = model_es.forecast(steps=steps)
        plt.figure(figsize=(10,5))
        plt.plot(self.ts, label='Original')
        plt.plot(forecast_es, label='ES Forecast', color='green')
        plt.title('Exponential Smoothing Forecast')
        plt.legend()
        plt.show()
        return forecast_es

    def evaluate_forecast(self, actual, predicted):
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        sign_acc = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100
        print(f"\nForecast Evaluation Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Sign Accuracy: {sign_acc:.2f}%")
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'Sign Accuracy': sign_acc}


if __name__ == "__main__":
    # Example Usage
    data_path = '/Users/deepikanath/dnath796/Git/Financial_Models/Data_center/timeseries_data.csv'  # Update path if necessary
    model = ForecastingModel(
        data_path=data_path,
        date_col='Date',
        value_col='Value'
    )

    model.plot_series()

    if not model.test_stationarity():
        model.difference_series()

    model.plot_acf_pacf()
    model.fit_arima(order=(1,1,1))
    model.residual_diagnostics()
    forecasted = model.forecast(steps=10)

    # Example evaluation using last 10 points (if available)
    # actual_values = model.ts[-10:]
    # model.evaluate_forecast(actual_values, forecasted)

    # Exponential Smoothing Forecast
    model.exponential_smoothing(trend='add', seasonal=None)