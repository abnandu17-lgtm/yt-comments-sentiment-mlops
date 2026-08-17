from src.youtube_api import get_video_id, get_comments
from src.predict import predict_sentiment


def analyze_video(url):

    video_id = get_video_id(url)

    if video_id is None:
        raise ValueError("Invalid YouTube URL")

    comments = get_comments(
        video_id,
        max_comments=100
    )

    if not comments:
        raise ValueError(
            "No comments found for this video"
        )

    result = predict_sentiment(
        comments
    )

    return result


if __name__ == "__main__":

    url = input(
        "Enter YouTube video URL: "
    )

    try:

        result = analyze_video(url)

        print("\n===== VIDEO SENTIMENT ANALYSIS =====")

        print(
            f"Total comments: "
            f"{result['total_comments']}"
        )

        print(
            f"Sentiment counts: "
            f"{result['sentiment_counts']}"
        )

        print(
            f"Sentiment percentages: "
            f"{result['sentiment_percentages']}"
        )

    except Exception as e:

        print(
            f"Error: {e}"
        )