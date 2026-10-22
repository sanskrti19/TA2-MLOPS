from fastapi import FastAPI
from pydantic import BaseModel, Field
from urllib3 import request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(
title="Real-Time Sentiment Analysis API",
description="TA2: Local Kubernetes Deployment with Helm",
version="1.0.0",
)

analyzer = SentimentIntensityAnalyzer()

class SentimentRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=5000)

@app.get("/")
def home():
    return {
"message": "Sentiment Analysis API is running!",
"status": "healthy",
}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    scores = analyzer.polarity_scores(request.text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "text": request.text,
        "sentiment": sentiment,
        "confidence_score": round(abs(compound), 4),
        "scores": scores,
    }
   