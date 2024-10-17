import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Streamlit app title
st.title("Agglomerative Clustering App")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# User input for the number of clusters
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)

# User input for data generation or upload
option = st.sidebar.radio("Choose Data Source:", ('Generate Random Data', 'Upload CSV File'))

if option == 'Generate Random Data':
    num_samples = st.sidebar.slider("Number of Samples", min_value=10, max_value=500, value=100)
    # Generate random data
    data, _ = make_blobs(n_samples=num_samples, centers=n_clusters, random_state=42)
    data_df = pd.DataFrame(data, columns=['X', 'Y'])
    st.write("Generated Data:")
    st.write(data_df)
else:
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file)
        data_df = data_df.drop(['Genre'], axis=1)
        if data_df.shape[1] < 2:
            st.warning("Please ensure your data has at least two columns for clustering.")
            st.stop()
        st.write("Uploaded Data:")
        st.write(data_df)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Convert DataFrame to NumPy array
data = data_df.values

# Perform Agglomerative Clustering
clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
labels = clustering.fit_predict(data)

# Add the cluster labels to the DataFrame
data_df['Cluster'] = labels

# Display clustered data
st.subheader("Clustered Data")
st.write(data_df)

# Plot clusters
st.subheader("Cluster Plot")
fig, ax = plt.subplots()
scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_title("Agglomerative Clustering Results")
st.pyplot(fig)

