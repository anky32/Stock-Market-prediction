import numpy as np
import pandas as pd
from pmdarima import auto_arima
import joblib
import os

from src.data.pipeline import fetch_yahoo
from src.data.preprocess import time_split
from src.eval.metrics import mae, rmse


def fit_arima(train_close: pd.Series):
    """
    Auto-select ARIMA(p,d,q) on train close series.
    Seasonal=False for a simple baseline.
    """
    model = auto_arima(
        train_close,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False
    )
    return model


def main():
    symbol = "AAPL"  # change to your real ticker
    start = "2018-01-01"
    train_ratio = 0.8

    df = fetch_yahoo(symbol, start=start)
    train_df, test_df = time_split(df, train_ratio=train_ratio)

    y_train = train_df["close"].astype(float)
    y_test = test_df["close"].astype(float)

    # =========================
    # TRAIN ARIMA MODEL
    # =========================
    model = fit_arima(y_train)

    # =========================
    # SAVE TRAINED ARIMA MODEL
    # =========================
    os.makedirs("models_saved", exist_ok=True)
    joblib.dump(model, "models_saved/arima.pkl")
    print("✅ ARIMA model saved to models_saved/arima.pkl")

    # =========================
    # FORECAST ON TEST SET
    # =========================
    n = len(y_test)
    y_pred = model.predict(n_periods=n)

    # =========================
    # EVALUATION METRICS
    # =========================
    m_mae = mae(y_test.values, y_pred)
    m_rmse = rmse(y_test.values, y_pred)

    print("Symbol:", symbol)
    print("Train rows:", len(y_train), "Test rows:", len(y_test))
    print("ARIMA order (p,d,q):", model.order)
    print("MAE:", m_mae)
    print("RMSE:", m_rmse)

    # Quick sanity peek
    print("First 5 actual:", y_test.values[:5])
    print("First 5 pred  :", np.array(y_pred)[:5])


if __name__ == "__main__":
    main()

    # python src/models/arima_baseline.py