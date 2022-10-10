import os
from typing import Union

import praw
import prawcore


APP_ID = os.environ["REDDIT_APP_ID"]
API_KEY = os.environ["REDDIT_API_KEY"]
USER_AGENT = os.environ["REDDIT_USER_AGENT"]


def fetch_subreddit_comments(
    reddit: praw.Reddit, sub_name: str, comment_limit: int, post_limit: int
) -> tuple[Union[list[str], None], str]:
    """
    Fetches a set of comments from hot posts of the given subreddit. Only
    fetches top-level comments.

    Parameters
    - reddit : praw.Reddit
        A Reddit instance used to interact with the Reddit API.
    - sub_name : str
        A subreddit name to search.
    - comment_limit : int
        The maximum number of comments to retrieve across all posts searched.
    - post_limit : int
        The maximum number of posts to search.
    """
    subreddit = reddit.subreddit(sub_name)

    comment_list: list[str] = []
    n_comments: int = 0

    # Loop through the top post_limit posts in the subreddit, by "hot"
    try:
        for submission in subreddit.hot(limit=post_limit):
            submission.comment_sort = "controversial"
            for comment in submission.comments:
                comment_list.append(comment.body)
                n_comments += 1
                # If we reach our specified comment limit, return what we have
                if n_comments >= comment_limit:
                    return comment_list, "Successfully retrieved comments"
        return comment_list, "Successfully retrieved comments"
    except prawcore.exceptions.Redirect:
        # Subreddit does not exist
        return None, "Subreddit does not exist"
    except prawcore.exceptions.NotFound:
        # Subreddit name is invalid
        return None, "Invalid subreddit name"


if __name__ == "__main__":

    reddit = praw.Reddit(client_id=APP_ID, client_secret=API_KEY, user_agent=USER_AGENT)

    comments = fetch_subreddit_comments(reddit, "machinelearning", 5, 3)

    print(comments)

    with open("output.txt", "w") as f:
        for comment in comments:
            f.write(comment)
