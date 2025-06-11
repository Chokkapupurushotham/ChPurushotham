# ===================== Install Dependencies =====================
# Run this in terminal before running the script:
# pip install pandas matplotlib seaborn scikit-learn streamlit openpyxl

# ===================== Imports =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

# ===================== Load Your Local Datasets =====================
# Historical Stock Data
history = pd.read_excel("Coca-Cola_stock_history.xlsx")

# Company Info (if needed for report/display)
info = pd.read_csv("Coca-Cola_stock_info.csv")

# Preview
print(history.head())
print(info.head())

# =====================  Clean and Prepare Data =====================
history.dropna(subset=["Close"], inplace=True)
history.fillna(method='ffill', inplace=True)
history.fillna(0, inplace=True)

# Convert 'Date' to datetime if not already
history['Date'] = pd.to_datetime(history['Date'],utc=True)

# Feature Engineering
history['MA_20'] = history['Close'].rolling(window=20).mean()
history['MA_50'] = history['Close'].rolling(window=50).mean()
history['Daily_Return'] = history['Close'].pct_change()
history['Volatility'] = history['Daily_Return'].rolling(window=20).std()
history.dropna(inplace=True)

# =====================  EDA Plots =====================
plt.figure(figsize=(12, 6))
plt.plot(history['Date'], history['Close'], label='Close Price')
plt.plot(history['Date'], history['MA_20'], '--', label='MA 20')
plt.plot(history['Date'], history['MA_50'], '--', label='MA 50')
plt.title('Coca-Cola Stock with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(history.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



plt.figure(figsize=(10,6))
history['Daily_Return'].hist(bins=50)
plt.title("Daily Returns Distribution")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(10,6))
history['Volatility'].plot()
plt.title("Volatility over Time")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.show()


# =====================  Model Training =====================
features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
X = history[features]
y = history['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# =====================  Streamlit Web App =====================
st.title("Coca-Cola Stock Price Predictor")

st.subheader("Historical Chart")
st.line_chart(history.set_index('Date')[['Close', 'MA_20', 'MA_50']])

st.subheader("Latest Model Output")
st.success(f"Predicted Last Close Price from Model: ${y_pred[-1]:.2f}")