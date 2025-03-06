import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define stock tickers and time range
stocks = ["AAPL", "TSLA"]
start_date = "2020-01-01"
end_date = "2024-01-01"

# Fetch stock data with auto_adjust=True (default) for accurate analysis
data = yf.download(stocks, start=start_date, end=end_date)["Close"]

# Display first few rows
print("Stock Price Data Overview:")
print(data.head())

# Define moving average periods
short_window = 20  # 20-day moving average
long_window = 50  # 50-day moving average

# Calculate SMA & EMA
for stock in stocks:
    data[f"{stock}_SMA_20"] = data[stock].rolling(window=short_window).mean()
    data[f"{stock}_SMA_50"] = data[stock].rolling(window=long_window).mean()
    data[f"{stock}_EMA_20"] = data[stock].ewm(span=short_window, adjust=False).mean()

# Volatility (Rolling Standard Deviation)
for stock in stocks:
    data[f"{stock}_Volatility"] = data[stock].rolling(window=20).std()

# Normalize stock prices for comparison
normalized_data = data[stocks].apply(lambda x: x / x.iloc[0])

# ---------- PLOTTING & BUSINESS INSIGHTS ---------- #

# 1️⃣ Apple Stock Trends
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['AAPL'], label='AAPL Price', color='blue')
plt.plot(data.index, data['AAPL_SMA_20'], label='AAPL 20-day SMA', linestyle='dashed', color='orange')
plt.plot(data.index, data['AAPL_SMA_50'], label='AAPL 50-day SMA', linestyle='dashed', color='red')
plt.title('Apple Stock Price & Moving Averages (2020-2024)')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid()
plt.show()

# Insight:
print("Insight: If AAPL’s 20-day SMA is above the 50-day SMA, it signals an uptrend (bullish).")

# 2️⃣ Tesla Stock Trends
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['TSLA'], label='TSLA Price', color='blue')
plt.plot(data.index, data['TSLA_SMA_20'], label='TSLA 20-day SMA', linestyle='dashed', color='orange')
plt.plot(data.index, data['TSLA_SMA_50'], label='TSLA 50-day SMA', linestyle='dashed', color='red')
plt.title('Tesla Stock Price & Moving Averages (2020-2024)')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid()
plt.show()

# 3️⃣ Volatility Trends
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['AAPL_Volatility'], label='AAPL Volatility', color='purple')
plt.plot(data.index, data['TSLA_Volatility'], label='TSLA Volatility', color='green')
plt.title('Stock Price Volatility (Rolling 20-day Std Dev)')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid()
plt.show()

# Insight:
print("Insight: Higher volatility means riskier stock price fluctuations. TSLA shows greater volatility.")

# 4️⃣ Performance Comparison (Normalized Prices)
plt.figure(figsize=(14, 7))
plt.plot(normalized_data.index, normalized_data['AAPL'], label='AAPL', color='blue')
plt.plot(normalized_data.index, normalized_data['TSLA'], label='TSLA', color='red')
plt.title('Stock Performance Comparison (Normalized)')
plt.xlabel('Date')
plt.ylabel('Relative Price')
plt.legend()
plt.grid()
plt.show()

# Insight:
print("Insight: Normalized comparison shows which stock has outperformed relative to 2020 levels.")
