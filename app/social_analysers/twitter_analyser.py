import os
from typing import Union

import tweepy


TWITTER_BEARER_TOKEN = os.environ["TWITTER_BEARER_TOKEN"]


def fetch_twitter_user_comments(
    client: tweepy.Client, user: str, n_tweets: int
) -> tuple[Union[list[str], None], str]:
    """
    Fetch a number of tweets from a Twitter user. Can only fetch between 10 and 100
    tweets.

    Parameters:
        - client : tweepy.Client
            A tweepy client instance.
        - user : str
            A Twitter username/handle.
        - n_tweets : int
            The number of tweets to fetch. Must be between 10 and 100 (inclusive).

    Returns:
        - tuple[list[str] | None, str]
            A tuple.
            If the request was successful, the first element will be a list of tweet
            texts; if unsuccessful, it will be None. The string will be a status
            message saying whether or not the request was successful, and if not, why.
    """

    # max_results must be between 10 and 100
    if not 10 <= n_tweets <= 100:
        raise ValueError("n_tweets must be between 10 adn 100")

    # Query template string
    query_template = "from:{} -is:retweet"

    # Add the user into the query
    query = query_template.format(user)

    # Try to fetch the tweets
    try:
        tweets = client.search_recent_tweets(query=query, max_results=n_tweets)
    except tweepy.errors.BadRequest:
        # Invalid Twitter handle (could be too long too)
        return None, "Invalid Twitter handle"

    # User does not exist
    if tweets.data is None:
        return None, "User does not exist"

    # Get the text content from the tweets and put them in a list
    tweet_text_list = []
    for tweet in tweets.data:
        tweet_text_list.append(tweet.text)

    return tweet_text_list, "Successfully retrieved tweets"


if __name__ == "__main__":

    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

    print(fetch_twitter_user_comments(client, "Xemmypoo", 10))
