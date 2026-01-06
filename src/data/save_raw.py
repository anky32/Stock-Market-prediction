from pathlib import Path
from src.data.fetch import fetch_yahoo

def main():
    symbol = "AAPL"
    df = fetch_yahoo(symbol, start="2015-01-01")

    out_dir = Path("data_raw")
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / f"{symbol}_raw.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(df))

if __name__ == "__main__":
    main()
