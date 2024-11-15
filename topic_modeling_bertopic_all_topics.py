from bertopic import BERTopic
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
import hdbscan
import umap
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt


# Clears unused memory in the MPS cache
torch.mps.empty_cache()

# Load and preprocess the data (if not already loaded)
bbc_data = pd.read_csv('BBC_dataset/bbc-news-data.csv', sep='\t')
bbc_data.columns = ['Category', 'File', 'Title', 'Content']
bbc_data['Processed_Text'] = bbc_data['Title'] + ' ' + bbc_data['Content']

# Use preprocessed text if already available; otherwise, preprocess as needed
texts = bbc_data['Processed_Text'].tolist()

# Load the largest compatible embedding model for BERTopic
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Custom dimensionality reduction model (Example: PCA instead of UMAP)
# pca_model = PCA(n_components=5)

# Custom clustering model (Example: KMeans instead of HDBSCAN)
# kmeans_model = KMeans(n_clusters=10, random_state=42)

# Custom UMAP model for dimensionality reduction
umap_model = umap.UMAP(
    n_neighbors=15,       # Controls local versus global structure (higher values capture more global)
    n_components=5,       # Dimensionality of the reduced embedding space
    metric='cosine',      # Distance metric used by UMAP; 'cosine' works well for text embeddings
    random_state=42       # Ensures reproducibility
)

# Custom HDBSCAN model for clustering
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=10,  # Minimum cluster size, adjusts the granularity of clusters
    min_samples=1,        # Minimum number of points in a neighborhood for a point to be considered core
    metric='euclidean',   # Distance metric; 'euclidean'
    cluster_selection_method='eom',  # Method for selecting clusters
    prediction_data=True   # Enables prediction for new data points
)

# Initialize BERTopic with custom models
topic_model = BERTopic(
    language="english",
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
)

# Fit the model on the dataset
topics, probabilities = topic_model.fit_transform(texts)

# Display topics
topics = topic_model.get_topic_info()
topics.to_excel('topics.xlsx', index=False)

# Visualize the topic frequencies
fig1 = topic_model.visualize_barchart()
fig1.write_html("topic_barchart_all_topics.html")

# Visualize topic clusters
fig2 = topic_model.visualize_topics()
fig2.write_html("topic_clusters_all_topics.html")
