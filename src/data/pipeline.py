import pandas as pd
import yfinance as yf


def fetch_yahoo(symbol: str, start: str = "2018-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance and return a clean DataFrame with columns:
    date, open, high, low, close, adj_close, volume

    Handles cases where yfinance returns MultiIndex columns (e.g., Open/Close with ticker level),
    which can become names like open_aapl, high_aapl, etc.
    """
    symbol_clean = symbol.strip().upper()
    symbol_lower = symbol_clean.lower()

    df = yf.download(symbol_clean, start=start, end=end, auto_adjust=False, progress=False)

    # Case 1: MultiIndex columns (often happens when yfinance includes ticker as a second level)
    if isinstance(df.columns, pd.MultiIndex):
        # Convert ('Open','AAPL') -> 'open_aapl'
        df.columns = [
            f"{str(level0).strip().lower().replace(' ', '_')}_{str(level1).strip().lower().replace(' ', '_')}"
            for (level0, level1) in df.columns
        ]

    df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # If columns are like open_aapl/high_aapl/... normalize them
    suffix = f"_{symbol_lower}"
    rename_map = {}

    # Prefer to rename these if present
    for base in ["open", "high", "low", "close", "adj_close", "volume"]:
        col_with_suffix = base + suffix
        if col_with_suffix in df.columns:
            rename_map[col_with_suffix] = base

    # Apply renaming if needed
    if rename_map:
        df = df.rename(columns=rename_map)

    # Final sanity check: required columns must exist now
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns from Yahoo data: {missing}. "
            f"Got columns: {df.columns.tolist()}"
        )

    # Clean + sort
    df = df.dropna().sort_values("date").reset_index(drop=True)

    # Keep a consistent column order (adj_close may or may not exist, but usually does)
    ordered = ["date", "open", "high", "low", "close"]
    if "adj_close" in df.columns:
        ordered.append("adj_close")
    ordered.append("volume")

    # Include any extra columns at the end (rare, but safe)
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    return df


def main():
    symbol = "AAPL"
    df = fetch_yahoo(symbol, start="2020-01-01")

    print("Symbol:", symbol)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print(df.head(3))
    print(df.tail(3))


if __name__ == "__main__":
    main()
