import yfinance as yf

df = yf.download("AAPL", start="2020-01-01", progress=False)
print(df.head())
print("Rows:", len(df))
