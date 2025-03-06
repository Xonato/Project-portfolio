import yfinance as yf
import pandas as pd

# Define stock tickers and time range
stocks = ["AAPL", "TSLA"]
start_date = "2020-01-01"
end_date = "2024-01-01"

# Fetch stock data
data = yf.download(stocks, start=start_date, end=end_date)["Close"]

# Calculate moving averages
data['AAPL_SMA_20'] = data['AAPL'].rolling(window=20).mean()
data['TSLA_SMA_20'] = data['TSLA'].rolling(window=20).mean()

# Save to CSV
data.to_csv("stock_data_with_moving_averages.csv")
print("File saved: stock_data_with_moving_averages.csv")
