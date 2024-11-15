import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt


# Load the Sentiment140 dataset
columns = ["target", "id", "date", "flag", "user", "text"]
data = pd.read_csv("sentiment140/versions/2/training.1600000.processed.noemoticon.csv",
                   encoding='latin-1', names=columns)
# target=0=negative and target=4=positive
tweeter_data = data[["target", "text"]]
tweeter_data["target"] = tweeter_data["target"].map({0: "negative", 4: "positive"})


"""
Sentiment Analysis - Lexicon-Based Method
"""
lex_tweeter_data = tweeter_data
# Load custom VADER lexicon and stop_words
sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Define preprocessing function
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'\@\w+|\#', '', text)
    # Lowercase and remove stopwords
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return " ".join(words)


# Apply preprocessing to text column
tweeter_data["clean_text"] = tweeter_data["text"].apply(preprocess_text)


# Define a function to classify sentiment
def lexicon_sentiment(text):
    scores = sid.polarity_scores(text)
    # Classify as positive or negative
    return "positive" if scores['compound'] >= 0 else "negative"


# Apply sentiment analysis using the lexicon method
tweeter_data["predicted_sentiment"] = tweeter_data["clean_text"].apply(lexicon_sentiment)

# Check accuracy of predictions
tweeter_data["is_correct"] = tweeter_data["target"] == tweeter_data["predicted_sentiment"]
accuracy_counts = tweeter_data["is_correct"].value_counts(normalize=True) * 100

# Plot the comparison of correct vs incorrect predictions
plt.figure(figsize=(10, 8))
accuracy_counts.plot(kind="bar", color=["green", "red"], legend=False)
plt.title("Accuracy of Sentiment Predictions")
plt.xlabel("Prediction Accuracy")
plt.ylabel("Frequency")
plt.xticks([0, 1], ["Correct", "Incorrect"], rotation=0)
plt.show()








