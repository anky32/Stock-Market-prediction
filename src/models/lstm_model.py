import numpy as np
import pandas as pd
from tensorflow import keras
import os
import joblib

layers = keras.layers
models = keras.models

from src.data.pipeline import fetch_yahoo
from src.data.preprocess import time_split, scale_close, make_sequences
from src.eval.metrics import mae, rmse


def build_lstm(lookback: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(64),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def invert_scaled(scaler, arr_1d: np.ndarray) -> np.ndarray:
    """Convert scaled values back to real prices."""
    return scaler.inverse_transform(arr_1d.reshape(-1, 1)).ravel()


def main():
    symbol = "AAPL"       # change later if needed
    start = "2018-01-01"
    train_ratio = 0.8
    lookback = 60
    epochs = 10
    batch_size = 32

    # 1) Fetch
    df = fetch_yahoo(symbol, start=start)
    train_df, test_df = time_split(df, train_ratio=train_ratio)

    # 2) Scale close using TRAIN only
    train_scaled, test_scaled, scaler = scale_close(train_df, test_df)

    # 3) Make sequences (LSTM format)
    X_train, y_train = make_sequences(train_scaled, lookback)
    X_test, y_test = make_sequences(test_scaled, lookback)

    if len(X_test) == 0:
        raise ValueError(
            f"Test set too small for lookback={lookback}. "
            f"Reduce lookback or use more data."
        )

    # 4) Build + train
    model = build_lstm(lookback)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # =========================
    # SAVE TRAINED LSTM + SCALER
    # =========================
    os.makedirs("models_saved", exist_ok=True)

    model.save("models_saved/lstm.h5")
    joblib.dump(scaler, "models_saved/scaler.pkl")

    print("✅ LSTM model saved to models_saved/lstm.h5")
    print("✅ Scaler saved to models_saved/scaler.pkl")

    # 5) Predict (scaled)
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()

    # 6) Invert scaling back to price
    y_test_price = invert_scaled(scaler, y_test.ravel())
    y_pred_price = invert_scaled(scaler, y_pred_scaled)

    # 7) Evaluate
    m_mae = mae(y_test_price, y_pred_price)
    m_rmse = rmse(y_test_price, y_pred_price)

    print("Symbol:", symbol)
    print("Train rows:", len(train_df), "Test rows:", len(test_df))
    print("Lookback:", lookback, "Epochs:", epochs)
    print("LSTM Test points:", len(y_test_price))
    print("MAE:", m_mae)
    print("RMSE:", m_rmse)
    print("First 5 actual:", y_test_price[:5])
    print("First 5 pred  :", y_pred_price[:5])


if __name__ == "__main__":
    main()
