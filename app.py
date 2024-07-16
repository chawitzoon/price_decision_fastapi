from fastapi import FastAPI
import uvicorn
import numpy as np
from mlib.mlib import predict
from pydantic import BaseModel
from typing import List

app = FastAPI()

class PriceData(BaseModel):
    prices: List[float]


@app.get("/")
async def root():
    return {"message": "hello root route"}

@app.post("/predict_next_price")
async def predict_next_price(data: PriceData):
    if len(data.prices) != 7:
        return {"error": "7 lookback prices are required."}

    input_prices = np.array([data.prices])
    next_price_predicted = predict(input_prices)
    return {"result": round(float(next_price_predicted), 5)}


if __name__ == "__main__":
    uvicorn.run(app, port=8080)
