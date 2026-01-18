from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services import predict_arima, predict_lstm, predict_hybrid


# =========================
# REQUEST / RESPONSE SCHEMAS
# =========================

class PredictionRequest(BaseModel):
    symbol: str


class PredictionResponse(BaseModel):
    ARIMA: float
    LSTM: float
    Hybrid: float


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="Stock Market Prediction System",
    description="ARIMA, LSTM and Hybrid stock price prediction API",
    version="1.0"
)

# Allow frontend access (HTML/JS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# ROOT ENDPOINT (OPTIONAL)
# =========================

@app.get("/")
def root():
    return {"message": "Stock Market Prediction API is running"}


# =========================
# PREDICTION ENDPOINT
# =========================

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """
    Returns next-step stock price predictions
    using ARIMA, LSTM, and Hybrid models.
    """

    # NOTE:
    # symbol is accepted for future extensibility
    # current deployment uses pre-trained AAPL models

    arima_pred = predict_arima()
    lstm_pred = predict_lstm([arima_pred] * 60)
    hybrid_pred = predict_hybrid(arima_pred, lstm_pred)

    return {
        "ARIMA": arima_pred,
        "LSTM": lstm_pred,
        "Hybrid": hybrid_pred
    }
