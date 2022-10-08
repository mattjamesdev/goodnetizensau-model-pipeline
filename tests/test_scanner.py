import os

import praw

from app.reddit_scanner.scanner import fetch_comments


APP_ID = os.environ["REDDIT_APP_ID"]
API_KEY = os.environ["REDDIT_API_KEY"]
USER_AGENT = os.environ["REDDIT_USER_AGENT"]


class TestFetchComments:

    reddit = praw.Reddit(client_id=APP_ID, client_secret=API_KEY, user_agent=USER_AGENT)

    def test_return_type_1(self):
        """
        Tests that the function returns a list.
        """
        sub_name = "destiny2"
        comment_limit = 1
        post_limit = 1
        comment_list = fetch_comments(self.reddit, sub_name, comment_limit, post_limit)
        assert (
            type(comment_list) == list
        ), f"Expected return type 'list', but got {type(comment_list)}"

    def test_return_type_2(self):
        """
        Tests that the function returns a list of strings.
        """
        sub_name = "destiny2"
        comment_limit = 5
        post_limit = 1
        comment_list = fetch_comments(self.reddit, sub_name, comment_limit, post_limit)
        for comment in comment_list:
            assert (
                type(comment) == str
            ), f"Expected return type 'list', but got {type(comment)}"
