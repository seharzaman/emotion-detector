import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import folium
from folium.plugins import HeatMap
import random
import os

# Pre-download NLTK resources (outside the app)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Precompiled regular expressions
url_re = re.compile(r"http\S+|www\S+|https\S+|@\S+|#\S+")
hashtag_re = re.compile(r'\#')
non_alphabetic_re = re.compile(r'[^\w\s]')
stop_words = set(stopwords.words('english'))

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_tweet(tweet):
    tweet = url_re.sub('', tweet)
    tweet = hashtag_re.sub('', tweet)
    tweet = non_alphabetic_re.sub('', tweet)
    tweet = tweet.lower()
    tokens = word_tokenize(tweet)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# Load data
file_path = r"C:/Users/sabah/Downloads/archive (5)/Tweets.csv"
tweets_df = pd.read_csv(file_path)

# Preprocess text and add processed_text column
tweets_df['processed_text'] = tweets_df['text'].apply(preprocess_tweet)

# Mapping emotions
emotion_mapping = {'positive': 'positive', 'neutral': 'neutral', 'negative': 'negative'}
tweets_df['emotion'] = tweets_df['airline_sentiment'].map(emotion_mapping)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(tweets_df['processed_text'], tweets_df['emotion'], test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Sentiment analysis function
def analyze_sentiment(text):
    preprocessed_text = preprocess_tweet(text)
    text_tfidf = vectorizer.transform([preprocessed_text])
    sentiment = model.predict(text_tfidf)[0]
    return sentiment

# Heatmap preparation
if 'tweet_coord' not in tweets_df.columns or tweets_df['tweet_coord'].isna().all():
    def random_coordinates(num):
        return [[random.uniform(-90, 90), random.uniform(-180, 180)] for _ in range(num)]
    coordinates = random_coordinates(len(tweets_df))
    tweets_df['tweet_coord'] = coordinates
else:
    tweets_df['tweet_coord'] = tweets_df['tweet_coord'].apply(lambda x: eval(x) if pd.notna(x) else [None, None])

tweets_df = tweets_df.dropna(subset=['tweet_coord'])

heat_data = [[row['tweet_coord'][0], row['tweet_coord'][1], 1] for index, row in tweets_df.iterrows() if row['tweet_coord'][0] is not None and row['tweet_coord'][1] is not None]

m = folium.Map(location=[20, 0], zoom_start=2)
HeatMap(heat_data).add_to(m)
heatmap_path = 'heatmap.html'
m.save(heatmap_path)

# Streamlit UI
st.title("Tweet Sentiment Analysis and Heatmap")

tweet_input = st.text_input("Enter your tweet:")
if st.button("Analyze Sentiment"):
    if tweet_input:
        try:
            predicted_sentiment = analyze_sentiment(tweet_input)
            if predicted_sentiment == 'positive':
                st.success("Predicted Sentiment: Positive")
            elif predicted_sentiment == 'neutral':
                st.info("Predicted Sentiment: Neutral")
            else:
                st.error("Predicted Sentiment: Negative")
        except Exception as e:
            st.error(f"Error occurred during sentiment analysis: {e}")

# Display heatmap
if st.checkbox('Show Heatmap'):
    st.subheader("Tweet Locations Heatmap")
    st.markdown(f'<iframe src="{heatmap_path}" width="700" height="500"></iframe>', unsafe_allow_html=True)

# Clean up heatmap file
if os.path.exists(heatmap_path):
    os.remove(heatmap_path)