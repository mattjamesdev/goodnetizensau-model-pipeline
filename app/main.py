from fastapi import FastAPI
from pydantic import BaseModel

from app.model_pipeline import predict_pipeline

app = FastAPI()


# class TextIn(BaseModel):
#     text: str


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


@app.post("/classify-text/{text}", response_model=PredictionOut)
def classify_text(payload: str):
    prediction, harsh_words, probabilities = predict_pipeline(payload)
    prediction = int(prediction)
    probabilities = [float(val) for val in probabilities]

    return {
        "statusCode": 200,
        "bullying": prediction,
        "words": harsh_words,
        "probs": probabilities,
    }
