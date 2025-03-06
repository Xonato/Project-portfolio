import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define stock tickers and time range
stocks = ["AAPL", "TSLA"]
start_date = "2020-01-01"
end_date = "2024-01-01"

# Fetch stock data
data = yf.download(stocks, start=start_date, end=end_date, auto_adjust=False)["Adj Close"]


# Display first few rows
print(data.head())

# Define moving average periods
short_window = 20  # 20-day moving average
long_window = 50  # 50-day moving average

# Calculate SMA
data['AAPL_SMA_20'] = data['AAPL'].rolling(window=short_window).mean()
data['AAPL_SMA_50'] = data['AAPL'].rolling(window=long_window).mean()

data['TSLA_SMA_20'] = data['TSLA'].rolling(window=short_window).mean()
data['TSLA_SMA_50'] = data['TSLA'].rolling(window=long_window).mean()

# Display first few rows
print(data.head())

# Calculate EMA
data['AAPL_EMA_20'] = data['AAPL'].ewm(span=short_window, adjust=False).mean()
data['TSLA_EMA_20'] = data['TSLA'].ewm(span=short_window, adjust=False).mean()

# Plot stock prices with moving averages
plt.figure(figsize=(14, 7))

plt.plot(data.index, data['AAPL'], label='AAPL Price', color='blue')
plt.plot(data.index, data['AAPL_SMA_20'], label='AAPL 20-day SMA', linestyle='dashed', color='orange')
plt.plot(data.index, data['AAPL_SMA_50'], label='AAPL 50-day SMA', linestyle='dashed', color='red')

plt.title('Apple Stock Price & Moving Averages')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid()
plt.show()

# Similar plot for Tesla
plt.figure(figsize=(14, 7))

plt.plot(data.index, data['TSLA'], label='TSLA Price', color='blue')
plt.plot(data.index, data['TSLA_SMA_20'], label='TSLA 20-day SMA', linestyle='dashed', color='orange')
plt.plot(data.index, data['TSLA_SMA_50'], label='TSLA 50-day SMA', linestyle='dashed', color='red')

plt.title('Tesla Stock Price & Moving Averages')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid()
plt.show()

# Calculate rolling standard deviation (volatility)
data['AAPL_Volatility'] = data['AAPL'].rolling(window=20).std()
data['TSLA_Volatility'] = data['TSLA'].rolling(window=20).std()

# Plot volatility
plt.figure(figsize=(14, 7))

plt.plot(data.index, data['AAPL_Volatility'], label='AAPL Volatility', color='purple')
plt.plot(data.index, data['TSLA_Volatility'], label='TSLA Volatility', color='green')

plt.title('Stock Price Volatility (Rolling Std Dev)')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid()
plt.show()

# Normalize stock prices
normalized_data = data[['AAPL', 'TSLA']].apply(lambda x: x / x.iloc[0])

# Plot comparison
plt.figure(figsize=(14, 7))
plt.plot(normalized_data.index, normalized_data['AAPL'], label='AAPL', color='blue')
plt.plot(normalized_data.index, normalized_data['TSLA'], label='TSLA', color='red')

plt.title('Stock Performance Comparison')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid()
plt.show()