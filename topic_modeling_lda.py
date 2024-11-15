import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
bbc_data = pd.read_csv('BBC_dataset/bbc-news-data.csv', sep='\t')
bbc_data.columns = ['Category', 'File', 'Title', 'Content']

# Preprocess and combine Title and Content
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'\d+|[^\w\s]', '', text)  # Remove punctuation and numbers
    return ' '.join([word.lower() for word in text.split() if word.lower() not in stop_words])

bbc_data['Processed_Text'] = (bbc_data['Title'] + ' ' + bbc_data['Content']).apply(preprocess)

# Vectorize text
vectorizer = CountVectorizer(max_df=0.90, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(bbc_data['Processed_Text'])

# LDA Model
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
doc_topic_dist = lda_model.fit_transform(doc_term_matrix)
feature_names = vectorizer.get_feature_names_out()

# Visualize top words per topic
def plot_top_words(model, feature_names, n_top_words=10):
    fig, axes = plt.subplots(1, 5, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[-n_top_words:]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = topic[top_words_idx]
        ax = axes[topic_idx]
        ax.barh(top_words, top_weights)
        ax.set_title(f'Topic {topic_idx + 1}', fontsize=14, weight='bold')
        ax.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    plt.show()

plot_top_words(lda_model, feature_names)

# Topic Distribution Heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(doc_topic_dist, cmap="YlGnBu", cbar=True)
plt.xlabel("Topic", fontsize=14, weight='bold')
plt.ylabel("Document", fontsize=14, weight='bold')
plt.title("Document-Topic Distribution", fontsize=16, weight='bold')
plt.xticks(ticks=np.arange(5) + 0.5, labels=[1, 2, 3, 4, 5], fontsize=12, weight='bold')
plt.yticks(fontsize=12)
plt.show()

# Accuracy Checks
bbc_data['Predicted_Topic'] = doc_topic_dist.argmax(axis=1)
topic_to_category = {0: 'entertainment', 1: 'business', 2: 'politics', 3: 'tech', 4: 'sport'}
bbc_data['Predicted_Category'] = bbc_data['Predicted_Topic'].map(topic_to_category)
accuracy = accuracy_score(bbc_data['Category'], bbc_data['Predicted_Category'])

# Classification Report
classification_rep = classification_report(bbc_data['Category'], bbc_data['Predicted_Category'], target_names=bbc_data['Category'].unique())
print(classification_rep)

# Accuracy per Topic
bbc_data['Correct_Prediction'] = bbc_data['Category'] == bbc_data['Predicted_Category']
accuracy_per_topic = bbc_data.groupby('Category')['Correct_Prediction'].mean()

# Plot accuracy per topic
plt.figure(figsize=(10, 8))
sns.barplot(x=accuracy_per_topic.index, y=accuracy_per_topic.values)
plt.xlabel("Category", fontsize=14, weight='bold')
plt.ylabel("Accuracy", fontsize=14, weight='bold')
plt.title("Accuracy per Category", fontsize=16, weight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Precision and Recall per Category
precision = precision_score(bbc_data['Category'], bbc_data['Predicted_Category'], average=None, labels=bbc_data['Category'].unique())
recall = recall_score(bbc_data['Category'], bbc_data['Predicted_Category'], average=None, labels=bbc_data['Category'].unique())
unique_categories = bbc_data['Category'].unique()

# Plot Precision and Recall
plt.figure(figsize=(10, 8))
sns.barplot(x=unique_categories, y=precision)
plt.xlabel('Category', fontsize=14, weight='bold')
plt.ylabel('Precision', fontsize=14, weight='bold')
plt.title('Precision per Category', fontsize=16, weight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.barplot(x=unique_categories, y=recall)
plt.xlabel('Category', fontsize=14, weight='bold')
plt.ylabel('Recall', fontsize=14, weight='bold')
plt.title('Recall per Category', fontsize=16, weight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




