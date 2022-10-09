import logging
import os

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from praw import Reddit

from app.model_pipeline import predict_pipeline, analyse_comments
from app.reddit_scanner.scanner import fetch_comments


APP_ID = os.environ["REDDIT_APP_ID"]
API_KEY = os.environ["REDDIT_API_KEY"]
USER_AGENT = os.environ["REDDIT_USER_AGENT"]


app = FastAPI()

origins = ["https://goodnetizensau.ml"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextIn(BaseModel):
    input_text: str


class TextPredictionOut(BaseModel):
    bullying: int
    words: list[str]
    probs: list[float]


class SubredditPredictionOut(BaseModel):
    toxicity: float
    probs: list[float]


@app.get("/")
def index():
    return {
        "msg": "You've found the index! Head to <url>/docs for documentation.",
    }


# For logging more details in case of a 422 error
# https://github.com/tiangolo/fastapi/issues/3361#issuecomment-1002120988
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logging.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@app.post("/classify-text", response_model=TextPredictionOut)
def classify_text(payload: TextIn):
    text = payload.input_text
    prediction, harsh_words, probabilities = predict_pipeline(text)
    prediction = int(prediction)
    probabilities = [float(val) for val in probabilities]

    return {
        "bullying": prediction,
        "words": harsh_words,
        "probs": probabilities,
    }


@app.post("/analyse-subreddit", response_model=SubredditPredictionOut)
def analyse_subreddit(payload: TextIn):
    subreddit_name = payload.input_text

    reddit = Reddit(client_id=APP_ID, client_secret=API_KEY, user_agent=USER_AGENT)

    n_comments = 100
    post_limit = 5

    # Get comments from the subreddit
    comments = fetch_comments(
        reddit, subreddit_name, comment_limit=n_comments, post_limit=post_limit
    )

    # Analyse the comments to get a toxicity fraction and a proportion of
    # probabilities for each category (toxic, aggressive, attacking)
    fraction_toxic, class_probabilities = analyse_comments(comments)

    return {"toxicity": fraction_toxic, "probs": class_probabilities}
