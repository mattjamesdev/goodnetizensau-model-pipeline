import os

from fastapi import FastAPI
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
