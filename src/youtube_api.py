import os
import re

from dotenv import load_dotenv
from googleapiclient.discovery import build


# Load environment variables
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")


def get_video_id(url):

    # Extract video ID from normal YouTube URL
    pattern = r"(?:v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"

    match = re.search(
        pattern,
        url
    )

    if match:
        return match.group(1)

    return None


def get_comments(video_id, max_comments=100):

    if not API_KEY:
        raise Exception(
            "YouTube API key not found"
        )

    youtube = build(
        "youtube",
        "v3",
        developerKey=API_KEY
    )

    comments = []

    next_page_token = None

    while len(comments) < max_comments:

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(
                100,
                max_comments - len(comments)
            ),
            pageToken=next_page_token,
            textFormat="plainText"
        )

        response = request.execute()

        for item in response.get(
            "items",
            []
        ):

            comment = item[
                "snippet"
            ][
                "topLevelComment"
            ][
                "snippet"
            ][
                "textDisplay"
            ]

            comments.append(
                comment
            )

        next_page_token = response.get(
            "nextPageToken"
        )

        if not next_page_token:
            break

    return comments


if __name__ == "__main__":

    url = input(
        "Enter YouTube video URL: "
    )

    video_id = get_video_id(
        url
    )

    if video_id is None:

        print(
            "Invalid YouTube URL"
        )

    else:

        comments = get_comments(
            video_id
        )

        print(
            f"Video ID: {video_id}"
        )

        print(
            f"Comments fetched: {len(comments)}"
        )

        for comment in comments[:5]:

            print(
                comment
            )