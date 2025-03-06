import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("stock_data_with_moving_averages.csv", index_col=0, parse_dates=True)

# Convert dates to numeric format
data['Days'] = (data.index - data.index[0]).days  

# Prepare data for prediction (using AAPL)
X = data[['Days']]
y = data['AAPL']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future prices
future_days = np.arange(X_test['Days'].max() + 1, X_test['Days'].max() + 31).reshape(-1, 1)
future_prices = model.predict(future_days)

# Plot predictions
plt.figure(figsize=(14, 7))
plt.scatter(X_train, y_train, label='Training Data', color='blue')
plt.scatter(X_test, y_test, label='Test Data', color='red')
plt.plot(future_days, future_prices, label='Future Prediction', color='green', linestyle='dashed')

plt.title('Stock Price Prediction (AAPL)')
plt.xlabel('Days Since Start')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid()
plt.show()
