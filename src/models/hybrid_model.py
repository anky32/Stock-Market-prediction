import numpy as np
import pandas as pd
from pmdarima import auto_arima
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

from src.data.pipeline import fetch_yahoo
from src.data.preprocess import time_split
from src.eval.metrics import mae, rmse

layers = keras.layers
models = keras.models


def make_sequences(arr: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback:i])
        y.append(arr[i])
    X = np.array(X, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


def build_lstm(lookback: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(64),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    symbol = "AAPL"
    start = "2018-01-01"
    train_ratio = 0.8

    lookback = 60
    epochs = 10
    batch_size = 32

    # 1) Fetch + split
    df = fetch_yahoo(symbol, start=start)
    train_df, test_df = time_split(df, train_ratio=train_ratio)

    y_train = train_df["close"].astype(float)
    y_test = test_df["close"].astype(float)
    horizon = len(y_test)

    # 2) Fit ARIMA on train close
    arima = auto_arima(
        y_train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False
    )

    # 3) In-sample fitted values for residuals
    fitted = pd.Series(arima.predict_in_sample(), index=y_train.index)

    # Align lengths safely (sometimes predict_in_sample can be shorter due to differencing)
    y_train_aligned = y_train.loc[fitted.index]
    resid = (y_train_aligned - fitted).dropna()

    # 4) Scale residuals (train only)
    scaler = MinMaxScaler(feature_range=(0, 1))
    resid_scaled = scaler.fit_transform(resid.values.reshape(-1, 1)).ravel()

    # 5) Prepare LSTM sequences on residuals
    Xr, yr = make_sequences(resid_scaled, lookback)
    if len(Xr) == 0:
        raise ValueError("Residual series too short for lookback. Reduce lookback or fetch more data.")

    lstm = build_lstm(lookback)
    lstm.fit(Xr, yr, epochs=epochs, batch_size=batch_size, verbose=1)

    # 6) Forecast ARIMA for test horizon
    arima_fc = arima.predict(n_periods=horizon)

    # 7) Forecast residuals iteratively for horizon
    seq = resid_scaled[-lookback:].tolist()
    resid_pred_scaled = []
    for _ in range(horizon):
        x = np.array(seq[-lookback:], dtype=np.float32).reshape(1, lookback, 1)
        yhat = float(lstm.predict(x, verbose=0).ravel()[0])
        resid_pred_scaled.append(yhat)
        seq.append(yhat)

    resid_pred = scaler.inverse_transform(np.array(resid_pred_scaled).reshape(-1, 1)).ravel()

    # 8) Hybrid prediction = ARIMA + residual forecast
    hybrid_pred = np.array(arima_fc, dtype=float) + resid_pred

    # 9) Evaluate
    m_mae = mae(y_test.values, hybrid_pred)
    m_rmse = rmse(y_test.values, hybrid_pred)

    print("Symbol:", symbol)
    print("Train rows:", len(y_train), "Test rows:", len(y_test))
    print("ARIMA order (p,d,q):", arima.order)
    print("Lookback:", lookback, "Epochs:", epochs)
    print("Hybrid MAE:", m_mae)
    print("Hybrid RMSE:", m_rmse)
    print("First 5 actual:", y_test.values[:5])
    print("First 5 pred  :", hybrid_pred[:5])


if __name__ == "__main__":
    main()
