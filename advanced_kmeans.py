import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px

# Set the Streamlit app title
st.title("Advanced Real-Time K-Means Clustering App")

# Sidebar settings for data upload and clustering parameters
st.sidebar.header("Settings")

# Upload CSV data
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Number of clusters
num_clusters = st.sidebar.slider("Number of Clusters", 1, 10, 3)

# Choose dimensionality reduction method
dimensionality_reduction = st.sidebar.selectbox("Dimensionality Reduction Method", ["None", "PCA", "t-SNE"])

# Read and preprocess data
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
    #data = X.iloc[:, [3, 4]].values
    data = data.drop(['Genre'], axis=1)
    # Feature selection
    features = st.sidebar.multiselect("Select features for clustering", data.columns.tolist(), default=data.columns.tolist())
    
    # Standardize the data
    X = data[features].values
    X = StandardScaler().fit_transform(X)

    # Apply dimensionality reduction if selected
    if dimensionality_reduction == "PCA":
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        st.write(f"Explained Variance by PCA: {np.sum(pca.explained_variance_ratio_):.2%}")
    elif dimensionality_reduction == "t-SNE":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        X_reduced = tsne.fit_transform(X)
    else:
        X_reduced = X[:, :2]  # Use the first two features for visualization if no reduction

    # K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    # Visualize clusters using Plotly
    df_plot = pd.DataFrame(X_reduced, columns=['Component 1', 'Component 2'])
    df_plot['Cluster'] = y_kmeans
    df_plot['Original Index'] = data.index

    fig = px.scatter(df_plot, x='Component 1', y='Component 2', color='Cluster',
                     title="K-Means Clustering Visualization",
                     hover_data=['Original Index'], opacity=0.7)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig)

    # Display cluster centroids if data is not reduced
    if dimensionality_reduction == "None":
        centroids = kmeans.cluster_centers_[:, :2]
        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', edgecolor='black', label='Centroids')
        st.pyplot(plt)
    
    # Option to display the original data with assigned clusters
    if st.checkbox("Show Clustered Data"):
        clustered_data = data.copy()
        clustered_data['Cluster'] = y_kmeans
        st.write(clustered_data)
else:
    st.write("Please upload a CSV file to proceed.")
