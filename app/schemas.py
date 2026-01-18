from pydantic import BaseModel

class PredictionRequest(BaseModel):
    symbol: str

class PredictionResponse(BaseModel):
    model: str
    prediction: float
