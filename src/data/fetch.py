import pandas as pd
import yfinance as yf

def _clean_columns(cols) -> list[str]:
    cleaned = []
    for c in cols:
        # If column name is a tuple (MultiIndex), flatten it
        if isinstance(c, tuple):
            # Keep non-empty parts and join with underscore
            c = "_".join([str(x) for x in c if x not in (None, "", " ")])
        else:
            c = str(c)

        c = c.strip().lower().replace(" ", "_")
        cleaned.append(c)
    return cleaned

def fetch_yahoo(symbol: str, start="2018-01-01", end=None) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)

    # Sometimes yfinance returns a MultiIndex columns. Flatten safely.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x]) for col in df.columns]

    df = df.reset_index()
    df.columns = _clean_columns(df.columns)

    return df

if __name__ == "__main__":
    df = fetch_yahoo("AAPL", start="2020-01-01")
    print(df.head())
    print(df.tail())
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
