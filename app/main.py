from fastapi import FastAPI
from pydantic import BaseModel

from app.model_pipeline import predict_pipeline

app = FastAPI()


class TextIn(BaseModel):
    input_text: str


class PredictionOut(BaseModel):
    bullying: int
    words: list[str]
    probs: list[float]


@app.get("/")
def index():
    return {
        "statusCode": 200,
        "msg": "You've found the index! Head to <url>/docs for documentation.",
    }


@app.post("/classify-text", response_model=PredictionOut)
def classify_text(payload: TextIn):
    text = payload.input_text
    prediction, harsh_words, probabilities = predict_pipeline(text)
    prediction = int(prediction)
    probabilities = [float(val) for val in probabilities]

    return {
        "statusCode": 200,
        "bullying": prediction,
        "words": harsh_words,
        "probs": probabilities,
    }
