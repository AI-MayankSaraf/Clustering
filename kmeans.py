import streamlit as st
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set up the Streamlit app title
st.title("K-Means Clustering Visualization")

# Sidebar settings for the number of clusters and dataset size
st.sidebar.header("Settings")
num_clusters = st.sidebar.slider("Number of Clusters", 1, 10, 3)
num_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)

# Generate synthetic data using make_blobs
X, _ = make_blobs(n_samples=num_samples, centers=num_clusters, cluster_std=1.0, random_state=42)

# K-Means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting the clusters and centroids
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
ax.set_title("K-Means Clustering")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

# Display the plot
st.pyplot(fig)

# Option to show data
if st.checkbox("Show Data"):
    st.write(X)
