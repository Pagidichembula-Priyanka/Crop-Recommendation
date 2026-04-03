# Crop Recommendation System (MLOps Project)

## Features
- ML model using Random Forest
- MLflow experiment tracking
- FastAPI for prediction
- Dockerized deployment

## Run locally
```bash
python src/train.py
uvicorn src.main:app --reload

Run with Docker
docker build -t crop-api .
docker run -p 8000:8000 crop-api