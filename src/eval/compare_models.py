import numpy as np
import matplotlib.pyplot as plt

from src.models.arima_baseline import fit_arima
from src.models.lstm_model import build_lstm, invert_scaled
from src.models.hybrid_model import build_lstm as build_resid_lstm, make_sequences as make_resid_sequences
from src.data.pipeline import fetch_yahoo
from src.data.preprocess import time_split, scale_close, make_sequences
from src.eval.metrics import mae, rmse
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def run_arima(y_train, y_test):
    model = fit_arima(y_train)
    pred = model.predict(n_periods=len(y_test))
    return np.asarray(pred, dtype=float), model.order


def run_lstm(train_df, test_df, lookback=60, epochs=10, batch_size=32):
    train_scaled, test_scaled, scaler = scale_close(train_df, test_df)

    X_train, y_train = make_sequences(train_scaled, lookback)
    X_test, y_test = make_sequences(test_scaled, lookback)

    model = build_lstm(lookback)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    pred_scaled = model.predict(X_test, verbose=0).ravel()

    y_true = invert_scaled(scaler, y_test.ravel())
    y_pred = invert_scaled(scaler, pred_scaled)

    return y_true, y_pred


def run_hybrid(y_train, y_test, lookback=60, epochs=10, batch_size=32):
    horizon = len(y_test)

    arima = auto_arima(
        y_train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False
    )

    fitted = pd.Series(arima.predict_in_sample(), index=y_train.index)
    y_train_aligned = y_train.loc[fitted.index]
    resid = (y_train_aligned - fitted).dropna()

    scaler = MinMaxScaler()
    resid_scaled = scaler.fit_transform(resid.values.reshape(-1, 1)).ravel()

    Xr, yr = make_resid_sequences(resid_scaled, lookback)
    lstm = build_resid_lstm(lookback)
    lstm.fit(Xr, yr, epochs=epochs, batch_size=batch_size, verbose=0)

    arima_fc = np.asarray(arima.predict(n_periods=horizon), dtype=float)

    seq = resid_scaled[-lookback:].tolist()
    resid_pred_scaled = []
    for _ in range(horizon):
        x = np.array(seq[-lookback:], dtype=np.float32).reshape(1, lookback, 1)
        yhat = float(lstm.predict(x, verbose=0).ravel()[0])
        resid_pred_scaled.append(yhat)
        seq.append(yhat)

    resid_pred = scaler.inverse_transform(np.array(resid_pred_scaled).reshape(-1, 1)).ravel()
    hybrid_pred = arima_fc + resid_pred

    return hybrid_pred, arima.order


def main():
    symbol = "AAPL"
    start = "2018-01-01"
    train_ratio = 0.8
    lookback = 60
    epochs = 10

    df = fetch_yahoo(symbol, start=start)
    train_df, test_df = time_split(df, train_ratio=train_ratio)

    y_train = train_df["close"].astype(float)
    y_test = test_df["close"].astype(float).values

    # ARIMA
    arima_pred, arima_order = run_arima(y_train, pd.Series(y_test))

    # LSTM (note: it predicts fewer test points because of lookback)
    lstm_true, lstm_pred = run_lstm(train_df, test_df, lookback=lookback, epochs=epochs)

    # HYBRID
    hybrid_pred, hybrid_order = run_hybrid(y_train, pd.Series(y_test), lookback=lookback, epochs=epochs)

    # Metrics
    arima_mae, arima_rmse = mae(y_test, arima_pred), rmse(y_test, arima_pred)
    hybrid_mae, hybrid_rmse = mae(y_test, hybrid_pred), rmse(y_test, hybrid_pred)
    lstm_mae, lstm_rmse = mae(lstm_true, lstm_pred), rmse(lstm_true, lstm_pred)

    print("\n=== Model Comparison (Yahoo Finance) ===")
    print("Symbol:", symbol)
    print(f"ARIMA {arima_order}  -> MAE: {arima_mae:.4f}  RMSE: {arima_rmse:.4f}")
    print(f"LSTM (lookback={lookback}) -> MAE: {lstm_mae:.4f}  RMSE: {lstm_rmse:.4f}")
    print(f"Hybrid ARIMA{hybrid_order}+LSTM(resid) -> MAE: {hybrid_mae:.4f}  RMSE: {hybrid_rmse:.4f}")

    # Plot (use aligned lengths for fair plot on same chart)
    plt.figure()
    plt.title(f"{symbol} - Actual vs Predictions")
    plt.plot(y_test, label="Actual (Test)")

    plt.plot(arima_pred, label="ARIMA")
    plt.plot(hybrid_pred, label="Hybrid")

    # LSTM has shorter series (because lookback)
    offset = len(y_test) - len(lstm_pred)
    plt.plot(range(offset, offset + len(lstm_pred)), lstm_pred, label="LSTM")

    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/comparison.png", dpi=300)
    print("\nSaved plot: comparison.png")


if __name__ == "__main__":
    main()
