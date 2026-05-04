# install first:
# pip install praw pandas python-dotenv

from gettext import install
import os
import pip
import praw
import pandas as pd
from datetime import datetime, time, timezone
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import re

# --------------------------------------------------
# 1. Set working directory
# --------------------------------------------------

os.chdir("/Users/songtao/Dropbox/26SP/SODA 501/soda501_ps/final project")
os.makedirs("data_raw", exist_ok=True)
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

# --------------------------------------------------
# 2. Load Reddit API credentials
# --------------------------------------------------
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# --------------------------------------------------
# 3. Define subreddits and keywords
# --------------------------------------------------
subreddits = [
    "Pennsylvania",
    "Pittsburgh",
    "philadelphia",
    "PAWilds",
    "Harrisburg",
    "Scranton",
    "LehighValley"
]

keywords = [
    "data center",
    "data centers",
    "server farm",
    "server farms",
    "AI data center",
    "cloud computing",
    "warehouse data center"
]

# --------------------------------------------------
# 4. Collect posts and comments from past 12 months
# --------------------------------------------------
post_rows = []
comment_rows = []

for sub in subreddits:
    subreddit = reddit.subreddit(sub)

    for keyword in keywords:
        print(f"Searching r/{sub} for: {keyword}")

        for post in subreddit.search(
            query=f'"{keyword}"',
            sort="new",
            time_filter="year",
            limit=200
        ):
            post_rows.append({
                "subreddit": sub,
                "keyword": keyword,
                "post_id": post.id,
                "post_title": post.title,
                "post_text": post.selftext,
                "post_author": str(post.author),
                "post_score": post.score,
                "post_upvote_ratio": post.upvote_ratio,
                "post_num_comments": post.num_comments,
                "post_created_utc": datetime.fromtimestamp(
                    post.created_utc, tz=timezone.utc
                ),
                "post_url": post.url,
                "post_permalink": "https://www.reddit.com" + post.permalink
            })

            post.comments.replace_more(limit=None)

            for comment in post.comments.list():
                comment_rows.append({
                    "subreddit": sub,
                    "keyword": keyword,
                    "post_id": post.id,
                    "comment_id": comment.id,
                    "parent_id": comment.parent_id,
                    "comment_author": str(comment.author),
                    "comment_text": comment.body,
                    "comment_score": comment.score,
                    "comment_created_utc": datetime.fromtimestamp(
                        comment.created_utc, tz=timezone.utc
                    ),
                    "comment_permalink": "https://www.reddit.com" + comment.permalink
                })

# --------------------------------------------------
# 5. Convert to data frames
# --------------------------------------------------
posts_df = pd.DataFrame(post_rows)
comments_df = pd.DataFrame(comment_rows)

# Remove duplicate posts/comments caused by overlapping keywords
posts_df = posts_df.drop_duplicates(subset=["post_id"])
comments_df = comments_df.drop_duplicates(subset=["comment_id"])

# --------------------------------------------------
# 6. Basic text cleaning for storage only
# --------------------------------------------------
posts_df["post_title"] = posts_df["post_title"].fillna("").str.strip()
posts_df["post_text"] = posts_df["post_text"].fillna("").str.strip()

comments_df["comment_text"] = comments_df["comment_text"].fillna("").str.strip()

# --------------------------------------------------
# 7. Save structured text data
# --------------------------------------------------
posts_df.to_csv("data_raw/reddit_pa_data_center_posts.csv", index=False)
comments_df.to_csv("data_raw/reddit_pa_data_center_comments.csv", index=False)

print("Data collection complete.")
print(f"Number of posts collected: {len(posts_df)}")
print(f"Number of comments collected: {len(comments_df)}")


# ----------------------------------
# Reload text data for analysis
# ----------------------------------
posts = pd.read_csv("data_raw/reddit_pa_data_center_posts.csv")
comments = pd.read_csv("data_raw/reddit_pa_data_center_comments.csv")

posts["post_created_utc"] = pd.to_datetime(posts["post_created_utc"])
comments["comment_created_utc"] = pd.to_datetime(comments["comment_created_utc"])

comments["month"] = comments["comment_created_utc"].dt.to_period("M").astype(str)

print(posts.shape)
print(comments.shape)


# ----------------------------------
# Prepare text data (Preprocessing)
# ----------------------------------
comments["text"] = comments["comment_text"].fillna("").astype(str)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

comments["text_clean"] = comments["text"].apply(clean_text)

comments = comments[comments["text_clean"].str.len() > 0]


# ----------------------------------
# 1. Top term frequency
# ----------------------------------

from sklearn.feature_extraction.text import CountVectorizer

stop_words = "english"

vectorizer = CountVectorizer(
    stop_words=stop_words,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(comments["text_clean"])
terms = vectorizer.get_feature_names_out()
counts = X.sum(axis=0).A1

term_freq = pd.DataFrame({
    "term": terms,
    "count": counts
}).sort_values("count", ascending=False)

top_terms = term_freq.head(20)

plt.figure(figsize=(8, 6))
plt.barh(top_terms["term"][::-1], top_terms["count"][::-1])
plt.xlabel("Frequency")
plt.ylabel("Term")
plt.title("Top Terms in Reddit Comments")
plt.tight_layout()
plt.savefig("output/figures/top_terms_reddit_comments.png", dpi=300)
plt.show()



# ----------------------------------
# 2. LDA Topic Modeling
# ----------------------------------

from sklearn.decomposition import LatentDirichletAllocation

topic_vectorizer = CountVectorizer(
    stop_words="english",
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 1)
)

X_topic = topic_vectorizer.fit_transform(comments["text_clean"])
topic_terms = topic_vectorizer.get_feature_names_out()

lda = LatentDirichletAllocation(
    n_components=4,
    random_state=123,
    max_iter=20
)

lda.fit(X_topic)

def show_topics(model, feature_names, n_words=10):
    topic_rows = []

    for topic_id, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]

        topic_rows.append({
            "topic": topic_id + 1,
            "top_words": ", ".join(top_words)
        })

    return pd.DataFrame(topic_rows)

topics_df = show_topics(lda, topic_terms, n_words=10)
print(topics_df)

topics_df.to_csv("output/tables/reddit_lda_topics.csv", index=False)


# ----------------------------------
# 3. Topic prevalence over time
# ----------------------------------

topic_probs = lda.transform(X_topic)

comments["topic"] = topic_probs.argmax(axis=1) + 1

topic_month = (
    comments
    .groupby(["month", "topic"])
    .size()
    .reset_index(name="n_comments")
)

topic_month["share"] = (
    topic_month
    .groupby("month")["n_comments"]
    .transform(lambda x: x / x.sum())
)

plt.figure(figsize=(9, 6))

for topic in sorted(topic_month["topic"].unique()):
    temp = topic_month[topic_month["topic"] == topic]
    plt.plot(temp["month"], temp["share"], marker="o", label=f"Topic {topic}")

plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Share of Comments")
plt.title("Topic Prevalence Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("output/figures/topic_prevalence_over_time.png", dpi=300)
plt.show()


# ----------------------------------
# 4. Sentiment analysis with VADER
# ----------------------------------

# pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

comments["sentiment_score"] = comments["text"].apply(
    lambda x: analyzer.polarity_scores(str(x))["compound"]
)

def classify_sentiment(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

comments["sentiment"] = comments["sentiment_score"].apply(classify_sentiment)

sentiment_counts = comments["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["sentiment", "count"]

plt.figure(figsize=(6, 4))
plt.bar(sentiment_counts["sentiment"], sentiment_counts["count"])
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.title("Sentiment Distribution of Reddit Comments")
plt.tight_layout()
plt.savefig("output/figures/sentiment_distribution.png", dpi=300)
plt.show()

# ----------------------------------
# 5. Sentiment over time
# ----------------------------------

sentiment_month = (
    comments
    .groupby(["month", "sentiment"])
    .size()
    .reset_index(name="n_comments")
)

sentiment_month["share"] = (
    sentiment_month
    .groupby("month")["n_comments"]
    .transform(lambda x: x / x.sum())
)

plt.figure(figsize=(9, 6))

for sentiment in ["positive", "neutral", "negative"]:
    temp = sentiment_month[sentiment_month["sentiment"] == sentiment]
    plt.plot(temp["month"], temp["share"], marker="o", label=sentiment)

plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Share of Comments")
plt.title("Sentiment Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("output/figures/sentiment_over_time.png", dpi=300)
plt.show()


# ----------------------------------
# 6. word cloud
# ----------------------------------
# pip install wordcloud

from wordcloud import WordCloud

all_text = " ".join(comments["text_clean"])

wordcloud = WordCloud(
    width=1000,
    height=600,
    background_color="white",
    stopwords=None
).generate(all_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Reddit Comments")
plt.tight_layout()
plt.savefig("output/figures/reddit_wordcloud.png", dpi=300)
plt.show()