import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data.pipeline import fetch_yahoo


def time_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Time-series split (no shuffle)."""
    n = len(df)
    cut = int(n * train_ratio)
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test


def scale_close(train: pd.DataFrame, test: pd.DataFrame):
    """
    Fit scaler ONLY on train close, then transform both train and test.
    Returns: train_scaled, test_scaled, scaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train["close"].values.reshape(-1, 1)).ravel()
    test_scaled = scaler.transform(test["close"].values.reshape(-1, 1)).ravel()
    return train_scaled, test_scaled, scaler


def make_sequences(arr: np.ndarray, lookback: int):
    """
    Convert 1D array into LSTM sequences.
    X shape: (samples, lookback, 1)
    y shape: (samples, 1)
    """
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback:i])
        y.append(arr[i])
    X = np.array(X, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


def main():
    symbol = "TSLA"
    lookback = 60

    df = fetch_yahoo(symbol, start="2018-01-01")
    train_df, test_df = time_split(df, train_ratio=0.8)

    train_scaled, test_scaled, scaler = scale_close(train_df, test_df)

    X_train, y_train = make_sequences(train_scaled, lookback)
    X_test, y_test = make_sequences(test_scaled, lookback)

    print("Symbol:", symbol)
    print("Total rows:", len(df))
    print("Train rows:", len(train_df), "Test rows:", len(test_df))
    print("Lookback:", lookback)
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)


if __name__ == "__main__":
    main()
