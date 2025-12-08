import yfinance as yf
import pandas as pd

# Define the ticker symbol for The Cigna Group
ticker_symbol = 'CI'

# Define the period and interval
# period="max" gets all available data
# interval="1mo" specifies monthly data, "1d" specifies daily data.
data = yf.download(ticker_symbol, period="max", interval="1d")
data = data.dropna()
# Optional: Print the first and last few rows of the data
#print("Monthly stock data for Cigna Group (CI):")
#print(data.head())
#print(data.tail())

# save to CSV   
data.to_csv('data\\cigna_group_daily_stock_data.csv')

# also get monthly data
data = yf.download(ticker_symbol, period="max", interval="1mo")
data = data.dropna()
# save to CSV
data.to_csv('data\\cigna_group_monthly_stock_data.csv')

# pull market rate
# S&P 500 index ticker
ticker_symbol = '^GSPC'

# Download max available daily data
data = yf.download(ticker_symbol, period="max", interval="1d")
data = data.dropna()
# Save to CSV
data.to_csv('data\\snp500_daily_data.csv')

# also get monthly data
data = yf.download(ticker_symbol, period="max", interval="1mo")
data = data.dropna()
# Save to CSV
data.to_csv('data\\snp500_monthly_data.csv')

# Save to CSV
data.to_csv('data\\snp500_monthly_data.csv')    

