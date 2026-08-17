import streamlit as st

from src.analyze_video import analyze_video


st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)


st.title("📊 YouTube Sentiment Analysis")
st.write(
    "Analyze audience sentiment from YouTube video comments."
)

st.divider()


url = st.text_input(
    "🎥 Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)


analyze_button = st.button(
    "🔍 Analyze Video",
    type="primary"
)


if analyze_button:

    if not url:

        st.warning(
            "Please enter a YouTube video URL."
        )

    else:

        try:

            with st.spinner(
                "Fetching comments and analyzing sentiment..."
            ):

                result = analyze_video(url)

            total = result["total_comments"]
            counts = result["sentiment_counts"]
            percentages = result["sentiment_percentages"]

            positive = counts.get("1", 0)
            neutral = counts.get("0", 0)
            negative = counts.get("-1", 0)

            positive_pct = percentages.get("1", 0)
            neutral_pct = percentages.get("0", 0)
            negative_pct = percentages.get("-1", 0)

            # Determine overall sentiment
            if positive_pct > neutral_pct and positive_pct > negative_pct:
                overall = "Positive"
            elif negative_pct > positive_pct and negative_pct > neutral_pct:
                overall = "Negative"
            else:
                overall = "Neutral"

            st.success(
                f"Analysis completed — Overall sentiment: {overall}"
            )

            st.header("📋 Analysis Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Comments Analyzed",
                total
            )

            col2.metric(
                "Positive",
                f"{positive_pct}%"
            )

            col3.metric(
                "Neutral",
                f"{neutral_pct}%"
            )

            col4.metric(
                "Negative",
                f"{negative_pct}%"
            )

            st.divider()

            st.header("📊 Sentiment Distribution")

            chart_data = {
                "Sentiment": [
                    "Positive",
                    "Neutral",
                    "Negative"
                ],
                "Comments": [
                    positive,
                    neutral,
                    negative
                ]
            }

            st.bar_chart(
                chart_data,
                x="Sentiment",
                y="Comments"
            )

            st.divider()

            st.header("🔎 Detailed Analysis")

            st.write(
                f"Out of **{total} comments**, "
                f"**{positive} ({positive_pct}%)** were classified "
                f"as positive, **{neutral} ({neutral_pct}%)** "
                f"as neutral, and **{negative} ({negative_pct}%)** "
                f"as negative."
            )

            st.subheader(
                "Overall Audience Reaction"
            )

            if overall == "Positive":

                st.write(
                    "The majority of analyzed comments express "
                    "a positive reaction toward the video."
                )

            elif overall == "Negative":

                st.write(
                    "The majority of analyzed comments express "
                    "a negative reaction toward the video."
                )

            else:

                st.write(
                    "The audience reaction is relatively neutral, "
                    "with neutral comments forming the largest group."
                )

        except Exception as e:

            st.error(
                f"Analysis failed: {e}"
            )