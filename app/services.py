import joblib
import numpy as np
from tensorflow.keras.models import load_model

arima = joblib.load("models_saved/arima.pkl")
lstm = load_model("models_saved/lstm.h5", compile=False)
scaler = joblib.load("models_saved/scaler.pkl")

def predict_arima(steps=1):
    pred = arima.predict(n_periods=steps)
    return float(pred.iloc[0])


def predict_lstm(last_values):
    scaled = scaler.transform(np.array(last_values).reshape(-1,1))
    X = scaled[-60:].reshape(1,60,1)
    pred = lstm.predict(X, verbose=0)
    return float(scaler.inverse_transform(pred)[0][0])

def predict_hybrid(arima_pred, lstm_pred):
    return arima_pred + (lstm_pred - arima_pred)
