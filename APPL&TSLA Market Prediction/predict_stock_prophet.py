import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import yfinance as yf
from news_sentiment import get_stock_sentiment  # Import the sentiment function

# List of stocks to predict
stocks = ["AAPL", "TSLA"]

# Loop through each stock and predict
for stock in stocks:
    # Load stock data
    df = yf.download(stock, start="2020-01-01", end="2024-01-01")[['Close', 'Volume']]
    
    # Rename columns for Prophet
    df = df.reset_index()
    df.columns = ['ds', 'y', 'volume']
    
    # Get real-time sentiment score
    sentiment_score = get_stock_sentiment(stock)

    # Ensure sentiment is not missing
    if sentiment_score is None or pd.isna(sentiment_score):
        sentiment_score = 0  # Default neutral sentiment

    # Assign sentiment score to entire DataFrame
    df['sentiment'] = [sentiment_score] * len(df)  # Ensure same length

    # Initialize Prophet model
    model = Prophet()
    model.add_regressor('volume')  # Include trading volume
    model.add_regressor('sentiment')  # Include sentiment analysis

    # Train the model
    model.fit(df)

    # Create future dates for 1 year (365 days)
    future = model.make_future_dataframe(periods=365)

    # Ensure future DataFrame has required regressors
    future['volume'] = df['volume'].iloc[-1]  # Use last known volume
    future['sentiment'] = df['sentiment'].iloc[-1]  # Use latest sentiment score

    # Ensure future DataFrame has required regressors
    future['volume'] = df['volume'].iloc[-1]  # Use last known volume
    future['sentiment'] = df['sentiment'].iloc[-1]  # Use latest sentiment score

    # Generate forecast
    forecast = model.predict(future)

    # Plot actual vs. predicted stock prices
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], label=f'Actual {stock} Prices', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label=f'Prophet Prediction for {stock}', linestyle="dashed", color='red')

    plt.title(f'{stock} Stock Price Prediction (With Real-Time Sentiment)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.grid()
    plt.show()
