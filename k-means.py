import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Datasheet
from lib.datasheet import DataSheet

# --- get original centroids ---
def get_original_centroids(scaler, kmeans):
    # Get the centroid coordinates from the model (they are in the SCALED space)
    scaled_centroids = kmeans.cluster_centers_
    print("Scaled Centroid Coordinates:\n", scaled_centroids)
    original_centroids = scaler.inverse_transform(scaled_centroids)
    print("\nOriginal Scale Centroid Coordinates (Price, Rating):\n", original_centroids)

    return original_centroids

# --- Create Clustering ---
def clustering(n_clusters, data_scaled):
    # Run K-Means with the optimal number of clusters
    result = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    result.fit(data_scaled)
    return result

# --- Data visualization from clustering result ---
def v_clustering(n_clusters, df, clustering_result=None, scaler=None):
    df_result = df.copy()
    df_result['Cluster'] = clustering_result.labels_
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

    # Centroid
    original_centroids = get_original_centroids(scaler, clustering_result)

    # Plot the centroids on top of the scatter plot
    plt.scatter(
        x=original_centroids[:, 1],  # The second column is 'Rating'
        y=original_centroids[:, 0],  # The first column is 'Price'
        marker='*',  # Use a distinct marker (e.g., 'X')
        s=300,  # Make the marker size large
        c='red',  # Use a standout color like red
        edgecolor='black',  # Add a black border for visibility
        label='Centroids'  # Add a label for the legend
    )

    plt.title(f'Dress Clusters (K={n_clusters}) with Centroids based on Price vs. Rating', fontsize=18)
    plt.xlabel('Customer Rating', fontsize=14)
    plt.ylabel('Price Level (Encoded)', fontsize=14)

    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

# --- SSE Scoring Method ---
def v_sse(k_range, data_scaled):
    sse = []
    for k in k_range:
        kmeans = clustering(k, data_scaled)
        kmeans.fit(data_scaled)
        sse.append(kmeans.inertia_)

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
def v_silhouette(k_range, data_scaled):
    silhouette_scores = []

    for k in k_range:
        result = clustering(k, data_scaled)

        # Count silhouette score
        score = silhouette_score(data_scaled, result.labels_)
        silhouette_scores.append(score)
        print(f"For K={k}, THe Silhouette Score: {score:.4f}")

    # Plot Silhouette Scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Scores for Optimal K', fontsize=16)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()


# Set plot style for better visuals
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# The optimal k based on testing
optimal_k = 9

# Clustering starts at 2 groups, the number of samples (n) = 501
# Square root of n = sqrt(501) ≈ 22.38
k_range = range(2, 25)

# Datasheet file
dataset_file_path = 'assets/Attribute DataSet.xlsx'
# New Datasheet with clustering result
output_dataset_file_path = 'assets/dress-datasheet-k-means-clustering.csv'

scaler = StandardScaler()
ds = DataSheet(dataset_file_path, scaler)
ds.load()
ds.training()

# # Clustering all in ranges
# for k in k_range:
#     print(f"Clustering #: {k}")
#     clustering_result = clustering(k, ds.get_data_scaled())
#     v_clustering(k, ds.get_df_processed(), clustering_result, scaler)

# Create visualization of Elbow methods
v_sse(k_range, ds.get_data_scaled())

# Create visualization of Silhouette Scoring
v_silhouette(k_range, ds.get_data_scaled())

# Run clustering with optimal k
clustering_result = clustering(optimal_k, ds.get_data_scaled())

# Add the cluster labels to our processed (but not scaled) DataFrame
v_clustering(optimal_k, ds.get_df_processed(), clustering_result, scaler)

# Create new datasheet with cluster label
clusters = clustering_result.labels_
ds.save_clusters(clusters, output_dataset_file_path)
