import logging
import os
from typing import Union

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from praw import Reddit
from tweepy import Client

from app.schemas import (
    TextIn,
    TextPredictionOut,
    SubredditPredictionOut,
    TwitterUserPredictionOut,
    BadResponseOut,
)
from app.model_pipeline import predict_pipeline, analyse_comments
from app.social_analysers.subreddit_analyser import fetch_subreddit_comments
from app.social_analysers.twitter_analyser import fetch_twitter_user_comments


APP_ID = os.environ["REDDIT_APP_ID"]
API_KEY = os.environ["REDDIT_API_KEY"]
USER_AGENT = os.environ["REDDIT_USER_AGENT"]

TWITTER_BEARER_TOKEN = os.environ["TWITTER_BEARER_TOKEN"]

app = FastAPI()

origins = ["https://goodnetizensau.ml"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post(
    "/analyse-subreddit", response_model=Union[SubredditPredictionOut, BadResponseOut]
)
def analyse_subreddit(payload: TextIn, response: Response):
    subreddit_name = payload.input_text

    reddit = Reddit(client_id=APP_ID, client_secret=API_KEY, user_agent=USER_AGENT)

    n_comments = 100
    post_limit = 5

    # Get comments from the subreddit
    comments, status_string = fetch_subreddit_comments(
        reddit, subreddit_name, comment_limit=n_comments, post_limit=post_limit
    )

    # If the request yielded no results
    if comments is None:
        if status_string == "Invalid subreddit name":
            # Invalid subreddit name - send 400 bad request
            response.status_code = status.HTTP_400_BAD_REQUEST
        elif status_string == "Subreddit does not exist":
            # Sub does not exist - send 404 not found
            response.status_code = status.HTTP_404_NOT_FOUND
        return {"detail": status_string}

    # Analyse the comments to get a toxicity fraction and a proportion of
    # probabilities for each category (toxic, aggressive, attacking)
    fraction_toxic, class_probabilities = analyse_comments(comments)

    return {"toxicity": fraction_toxic, "probs": class_probabilities}


@app.post(
    "/analyse-twitter-user",
    response_model=Union[TwitterUserPredictionOut, BadResponseOut],
)
def analyse_twitter_user(payload: TextIn, response: Response):
    user_handle = payload.input_text

    client = Client(bearer_token=TWITTER_BEARER_TOKEN)

    n_tweets = 100

    tweets, status_string = fetch_twitter_user_comments(client, user_handle, n_tweets)

    # If the request yielded no results
    if tweets is None:
        if status_string == "Invalid Twitter handle":
            # Twitter handle is invalid - send 400 bad request
            response.status_code = status.HTTP_400_BAD_REQUEST
        elif status_string == "User does not exist":
            # Twitter user does not exist - send 404 not found
            response.status_code = status.HTTP_404_NOT_FOUND
        return {"detail": status_string}
    else:
        fraction_toxic, class_probabilities = analyse_comments(tweets)

        return {"toxicity": fraction_toxic, "probs": class_probabilities}
