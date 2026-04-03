from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI app
app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")


# Define input schema
class CropInput(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float


# Health check
@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running"}


# Prediction API
@app.post("/predict")
def predict_crop(data: CropInput):
    input_data = [[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]]

    prediction = model.predict(input_data)

    return {
        "recommended_crop": prediction[0]
    }