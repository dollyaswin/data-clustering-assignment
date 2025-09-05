# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scipy for dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

# Sklearn for clustering and scaling
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scipy for dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

# Sklearn for clustering and scaling
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# --- get original centroids ---
def get_original_centroids(scaler, kmeans):
    # Get the centroid coordinates from the model (they are in the SCALED space)
    scaled_centroids = kmeans.cluster_centers_
    print("Scaled Centroid Coordinates:\n", scaled_centroids)
    original_centroids = scaler.inverse_transform(scaled_centroids)
    print("\nOriginal Scale Centroid Coordinates (Price, Rating):\n", original_centroids)

    return original_centroids

# --- data visualiation from clustering result ---
def v_clustering(n_clusters, df_result, price_map, original_centroids=None):
    plt.figure(figsize=(14, 9))
    sns.scatterplot(
        data=df_result,
        x='Rating',
        y='Price',
        hue='Cluster',
        palette='viridis', # A visually appealing color palette
        s=501,           # Size of the points
        alpha=0.8        # Transparency of the points
    )

    # Plot the centroids on top of the scatter plot
    # plt.scatter(
    #     x=original_centroids[:, 1],  # The second column is 'Rating'
    #     y=original_centroids[:, 0],  # The first column is 'Price'
    #     marker='*',  # Use a distinct marker (e.g., 'X')
    #     s=300,  # Make the marker size large
    #     c='red',  # Use a standout color like red
    #     edgecolor='black',  # Add a black border for visibility
    #     label='Centroids'  # Add a label for the legend
    # )

    plt.title(f'Dress Clusters (K={n_clusters}) with Centroids based on Price vs. Rating', fontsize=18)
    plt.xlabel('Customer Rating', fontsize=14)
    plt.ylabel('Price Level (Encoded)', fontsize=14)

    # Use original price labels for y-axis ticks for better interpretation
    price_ticks = sorted(df_result['Price'].unique())
    price_labels = [key for key, val in sorted(price_map.items(), key=lambda item: item[1]) if val in price_ticks]
    #plt.yticks(ticks=price_ticks, labels=price_labels)

    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

    # --- Visualize the results ---
    # plt.figure(figsize=(14, 9))
    # sns.scatterplot(
    #     data=df_processed,
    #     x='Rating',
    #     y='Price',
    #     hue='Cluster',
    #     palette='viridis',
    #     s=150,
    #     alpha=0.8
    # )

    # plt.title('Agglomerative Clustering Results (K=4)', fontsize=18)
    # plt.xlabel('Customer Rating', fontsize=14)
    # plt.ylabel('Price Level (Encoded)', fontsize=14)
    #
    # # Use original price labels for y-axis ticks
    # price_ticks = sorted(df_processed['Price'].unique())
    # price_labels = [key for key, val in sorted(price_map.items(), key=lambda item: item[1]) if val in price_ticks]
    # plt.yticks(ticks=price_ticks, labels=price_labels)
    #
    # plt.legend(title='Cluster')
    # plt.grid(True)
    # plt.show()

# --- SSE Scoring Method ---
def v_sse(k_range, data_scale):
    sse = []
    for k in k_range:
        kmeans = clustering(k, data_scale)
        kmeans.fit(data_scale)
        sse.append(kmeans.inertia_)

    # --- Elbow Method (SSE) ---
    # Plot SSE
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K', fontsize=16)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

# --- Silhouette Scoring Method ---
def v_shilhouette(k_range, data_scale):
    silhouette_scores = []
    # for k in k_range:
    #     kmeans = clustering(k, data_scale)
    #     score = silhouette_score(data_scale, kmeans.labels_)
    #     silhouette_scores.append(score)
    #
    # # Plot Silhouette Scores
    # plt.figure(figsize=(10, 6))
    # plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
    # plt.title('Silhouette Scores for Optimal K', fontsize=16)
    # plt.xlabel('Number of Clusters (K)', fontsize=12)
    # plt.ylabel('Silhouette Score', fontsize=12)
    # plt.xticks(k_range)
    # plt.grid(True)
    # plt.show()

    for k in k_range:
        # Membuat model Agglomerative Clustering untuk setiap nilai k
        agg_cluster = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')

        # Melakukan fitting dan prediksi label klaster
        cluster_labels = agg_cluster.fit_predict(data_scale)

        # Menghitung silhouette score
        score = silhouette_score(data_scale, cluster_labels)
        silhouette_scores.append(score)
        print(f"For K={k}, THe Silhouette Score: {score:.4f}")

    # --- Membuat Grafik Silhouette Score ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Scores untuk Jumlah Klaster Optimal (Agglomerative)', fontsize=16)
    plt.xlabel('Jumlah Klaster (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

# --- Create Dendogram Visualization ---
def v_dendogram(data_scale, method='ward'):
    # --- Plot the dendrogram ---
    linked = linkage(data_scale, method=method)
    plt.figure(figsize=(15, 8))
    dendrogram(
        linked,
        orientation='top',
        labels=None,  # We hide labels as there are too many
        distance_sort='descending',
        show_leaf_counts=True
    )

    # Draw a line to suggest the optimal cut for 4 clusters
    plt.axhline(y=6, color='r', linestyle='--')

    plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=16)
    plt.xlabel('Data Points (or cluster size in parentheses)', fontsize=12)
    plt.ylabel('Euclidean Distance (Ward)', fontsize=12)
    plt.show()

# --- Training data ---
def training(df_processed, price_map):
    # 2.1 Clean 'Rating' column
    # Replace comma with dot and convert to numeric
    # df_processed['Rating'] = df_processed['Rating'].str.replace(',', '.').astype(float)
    df_processed['Rating'] = df_processed['Rating']

    # 2.2 Ordinal Encoding for 'Price'
    df_processed['Price'] = df_processed['Price'].map(price_map)

    # 2.3 Handle Missing Values
    # Replace the string 'null' with NumPy's NaN (Not a Number)
    df_processed.replace('null', np.nan, inplace=True)

    # Fill missing values with the mode (most frequent value) of each column
    for column in df_processed.columns:
        if df_processed[column].isnull().any():
            mode_value = df_processed[column].mode()[0]
            df_processed[column].fillna(mode_value, inplace=True)
            print(f"Filled missing values in '{column}' with mode: '{mode_value}'")

    return df_processed

# --- Encoding the training data ---
def encode(df_processed, categorical_features):
    # Use get_dummies for one-hot encoding
    df_encoded = pd.get_dummies(df_processed, columns=categorical_features, drop_first=True)

    # 2.5 Create the final training data matrix 'X'
    # Drop the identifier and the original target label
    X = df_encoded.drop(columns=['Dress_ID', 'Recommendation'])

    print("\nPreprocessing Complete.")
    print(f"Shape of the final training data (X): {X.shape}")
    print("\nColumns of the final training data:")
    print(X.columns)

    #return X

    # Clustering only on Price and Rating
    X_2d = df_processed[['Price', 'Rating']].copy()
    return X_2d

# --- Scaling ---
def scaling(scaller, data):
    X_scaled = scaler.fit_transform(data)

    print("\nFeature scaling complete.")
    return X_scaled

# --- Create Clustering ---
def clustering(n_clusters, data_scale):
    # Run K-Means with the optimal number of clusters
    # kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    # kmeans_final.fit(data_scale)
    # return kmeans_final

    # Initialize the Agglomerative Clustering model
    # We use n_clusters=4 based on our dendrogram analysis
    cluster = AgglomerativeClustering(n_clusters, metric='euclidean', linkage='ward')
    # Fit the model and get the cluster labels
    cluster_labels = cluster.fit_predict(data_scale)
    return cluster_labels

# --- Write New Datasheet With Cluster Label ---
def create_new_datasheet(cluster):
    df['Agglomerative_Cluster'] = cluster

    # # Save the final dataframe to a new CSV file
    df.to_csv(output_dataset_file_path, index=False, sep=';')

    print(f"\nSuccessfully saved the new dataset with cluster labels to '{output_dataset_file_path}'")
    print("\nFirst 5 rows of the new dataset:")
    print(df.head())

    # Add the new cluster labels from this analysis to the original dataframe
    # df['Agglomerative_Cluster'] = cluster

    # Save the dataframe to a new CSV file
    # output_filename_agg = 'Attribute_DataSet_with_Agglomerative_Clusters.csv'
    # df.to_csv(output_filename_agg, index=False, sep=';')

    print(f"\nSuccessfully saved the new dataset to '{output_dataset_file_path}'")



# Set plot style for better visuals
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# Clustering starts at 2 groups
# Number of samples (n) = 501
# Square root of n = sqrt(501) ≈ 22.38
k_range = range(2, 25)

# The optimal k based on testing
optimal_k = 10

# --- Load the data ---
dataset_file_path = 'assets/Attribute DataSet.xlsx'
output_dataset_file_path = 'assets/dress-datasheet-agglomerative-clustering.csv'
try:
    df = pd.read_excel(dataset_file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'Attribute DataSet 100.csv' not found. Please make sure the file is in the correct directory.")
    exit()

# --- Initial Inspection ---
print("\nFirst 5 rows of the dataset:")
print(df[['Dress_ID', 'Style', 'Price', 'Rating', 'Size', 'Season', 'NeckLine', 'SleeveLength']].head())

print("\nDataset Information (dtypes, non-null counts):")
df.info()

# Create a new DataFrame with only the relevant features
# We use the df_processed which has Price and Rating cleaned and numerical

# Identify categorical columns (excluding ID, Recommendation, and already processed ones)
categorical_features = ['Style', 'Size', 'Season', 'NeckLine', 'SleeveLength', 'waiseline',
                        'Material', 'FabricType', 'Decoration', 'Pattern Type']
# Define the order of the categories
price_map = {'Low': 1, 'low': 1, 'Average': 2, 'average': 2, 'Medium': 3, 'medium': 3, 'High': 4, 'high': 4, 'very-high': 5}

df_copy = df.copy()

# Training, encoding and scaling the data
df_processed = training(df_copy, price_map)
X = encode(df_processed, categorical_features)

scaler = StandardScaler()
data_scaled = scaling(scaler, X)
# v_dendogram(data_scaled)


# Clustering all in ranges
# for k in k_range:
#     print(f"Clustering #: {k}")
#     kmeans = clustering(k, data_scaled)
#     df_processed['Cluster'] = kmeans.labels_
#     v_clustering(k, df_processed, price_map, get_original_centroids(scaler, kmeans))

# Visualization of methods
# v_sse(k_range, data_scaled)
# v_shilhouette(k_range, data_scaled)

# Clustering with optimal k
clustering_result = clustering(optimal_k, data_scaled)

# Add the cluster labels to our processed (but not scaled) DataFrame
df_processed['Cluster'] = clustering_result
v_clustering(optimal_k, df_processed, price_map)

# Create new datasheet with cluster label
create_new_datasheet(clustering_result)