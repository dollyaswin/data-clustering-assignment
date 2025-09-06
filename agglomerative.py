import matplotlib.pyplot as plt
import seaborn as sns

# Scipy for dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

# Sklearn for clustering and scaling
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Datasheet
from lib.datasheet import DataSheet

# --- Create Clustering ---
def clustering(n_clusters, data_scale):
    # Initialize the Agglomerative Clustering model
    cluster = AgglomerativeClustering(n_clusters, metric='euclidean', linkage='ward')
    # Fit the model and get the cluster labels
    clustering_result = cluster.fit_predict(data_scale)

    return clustering_result

# --- Data visualization from clustering result ---
def v_clustering(n_clusters, df, clustering_result):
    df_result = df.copy()
    df_result['Cluster'] = clustering_result
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

    plt.title(f'Dress Clusters (K={n_clusters}) with Centroids based on Price vs. Rating', fontsize=18)
    plt.xlabel('Customer Rating', fontsize=14)
    plt.ylabel('Price Level (Encoded)', fontsize=14)

    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

# --- Silhouette Scoring Method ---
def v_silhouette(k_range, data_scale):
    silhouette_scores = []

    for k in k_range:
        result = clustering(k, data_scale)

        # Count silhouette score
        score = silhouette_score(data_scale, result)
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

# --- Create Dendrogram Visualization ---
def v_dendrogram(data_scale, method='ward'):
    linked = linkage(data_scale, method=method)
    plt.figure(figsize=(15, 8))
    dendrogram(
        linked,
        orientation='top',
        labels=None,  # We hide labels as there are too many
        distance_sort='descending',
        show_leaf_counts=True
    )

    plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=16)
    plt.xlabel('Data Points (or cluster size in parentheses)', fontsize=12)
    plt.ylabel('Euclidean Distance (Ward)', fontsize=12)
    plt.show()


# Set plot style for better visuals
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# The optimal k based on Silhouette
optimal_k = 10

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

# Create visualization of dendrogram
v_dendrogram(ds.get_data_scaled())

# Create visualization of methods
v_silhouette(k_range, ds.get_data_scaled())

# Run clustering with optimal k
clustering_result = clustering(optimal_k, ds.get_data_scaled())

# Add the cluster labels to our processed (but not scaled) DataFrame
v_clustering(optimal_k, ds.get_df(), clustering_result)

# Create new datasheet with cluster label
clusters = clustering_result
ds.save_clusters(clusters, output_dataset_file_path)