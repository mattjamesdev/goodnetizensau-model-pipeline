from pydantic import BaseModel


class TextIn(BaseModel):
    input_text: str


class TextPredictionOut(BaseModel):
    bullying: int
    words: list[str]
    probs: list[float]


class SubredditPredictionOut(BaseModel):
    toxicity: float
    probs: list[float]


class TwitterUserPredictionOut(BaseModel):
    toxicity: float
    probs: list[float]
