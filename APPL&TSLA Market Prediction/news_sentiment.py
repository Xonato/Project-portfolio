import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_stock_sentiment(stock):
    """
    Fetches real-time news headlines for a stock and returns an average sentiment score.
    """
    url = f"https://finance.yahoo.com/quote/{stock}/news?p={stock}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract headlines
    headlines = [h.text for h in soup.find_all("h3")][:5]  # Get the top 5 headlines

    if not headlines:
        return 0  # No headlines found

    # Perform sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(headline)["compound"] for headline in headlines]

    # Average sentiment score
    avg_sentiment = sum(scores) / len(scores)
    return avg_sentiment

# Test the function
if __name__ == "__main__":
    stock_symbol = "AAPL"  # You can change this to TSLA, MSFT, etc.
    sentiment_score = get_stock_sentiment(stock_symbol)
    print(f"Real-time sentiment score for {stock_symbol}: {sentiment_score}")
