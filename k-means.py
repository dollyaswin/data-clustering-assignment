# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# --- get original centroids ---
def get_original_centroids(scaler, kmeans):
    # Get the centroid coordinates from the model (they are in the SCALED space)
    scaled_centroids = kmeans.cluster_centers_
    print("Scaled Centroid Coordinates:\n", scaled_centroids)
    original_centroids = scaler.inverse_transform(scaled_centroids)
    print("\nOriginal Scale Centroid Coordinates (Price, Rating):\n", original_centroids)

    return original_centroids

# --- data visualiation from clustering result ---
def v_clustering(n_clusters, df_result, price_map, original_centroids):
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

    # Use original price labels for y-axis ticks for better interpretation
    price_ticks = sorted(df_result['Price'].unique())
    price_labels = [key for key, val in sorted(price_map.items(), key=lambda item: item[1]) if val in price_ticks]
    #plt.yticks(ticks=price_ticks, labels=price_labels)

    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

# --- SSE Scoring Method ---
def v_sse(k_range, data_scale):
    sse = []
    for k in k_range:
        kmeans = clustering(k, data_scale)
        kmeans.fit(data_scale)
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

# --- Training data ---
def training(df_processed, price_map):
    # Replace comma with dot and convert to numeric
    df_processed['Price']  = df_processed['Price'].map(price_map)

    # Replace the string 'null' with NumPy's NaN (Not a Number)
    df_processed.replace('null', np.nan, inplace=True)

    # Fill missing values with the mode (most frequent value) of each column
    for column in df_processed.columns:
        if df_processed[column].isnull().any():
            mode_value = df_processed[column].mode()[0]
            df_processed[column].fillna(mode_value, inplace=True)
            print(f"Filled missing values in '{column}' with mode: '{mode_value}'")

    return df_processed

# --- Scaling ---
def scaling(scaller, data):
    X_2d = data[['Price', 'Rating']].copy()
    X_scaled = scaler.fit_transform(X_2d)

    print("\nFeature scaling complete.")
    return X_scaled

# --- Create Clustering ---
def clustering(n_clusters, data_scale):
    # Initialize the Agglomerative Clustering model
    cluster = AgglomerativeClustering(n_clusters, metric='euclidean', linkage='ward')
    # Fit the model and get the cluster labels
    clustering_result = cluster.fit_predict(data_scale)

    return clustering_result

# --- Create Clustering ---
def clustering(n_clusters, data_scale):
    # Run K-Means with the optimal number of clusters
    clustering_result = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clustering_result.fit(data_scale)
    return clustering_result

# --- Write New Datasheet With Cluster Label ---
def create_new_datasheet(kmeans):
    df['Cluster'] = kmeans.labels_

    # Save the final dataframe to a new CSV file
    df.to_csv(output_dataset_file_path, index=False, sep=';')

    print(f"\nSuccessfully saved the new dataset with cluster labels to '{output_dataset_file_path}'")
    print("\nFirst 5 rows of the new dataset:")
    print(df.head())

# --- Load datasheet ---
def load_datasheet(dataset_file_path):
    try:
        df = pd.read_excel(dataset_file_path)
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("Error: 'Attribute DataSet.xlsx' not found. Please make sure the file is in the correct directory.")
        exit()

    # --- Initial Inspection ---
    print("\nFirst 5 rows of the dataset:")
    print(df[['Dress_ID', 'Style', 'Price', 'Rating', 'Size', 'Season', 'NeckLine', 'SleeveLength']].head())

    print("\nDataset Information (dtypes, non-null counts):")
    df.info()
    return df


# Set plot style for better visuals
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# Clustering starts at 2 groups, the number of samples (n) = 501
# Square root of n = sqrt(501) ≈ 22.38
k_range = range(2, 25)

# The optimal k based on testing
optimal_k = 5

# New Datasheet with clustering result
output_dataset_file_path = 'assets/dress-datasheet-k-means-clustering.csv'

dataset_file_path = 'assets/Attribute DataSet.xlsx'

# Load the datasheet
df = load_datasheet(dataset_file_path)
df_copy = df.copy()
scaler  = StandardScaler()

# Training and scaling the data
# Define the order of the categories
pricing_map  = {'Low': 1, 'low': 1, 'Average': 2, 'average': 2, 'Medium': 3, 'medium': 3, 'High': 4, 'high': 4, 'very-high': 5}
df_processed = training(df_copy, pricing_map)
data_scaled  = scaling(scaler, df_processed)

# Clustering all in ranges
# for k in k_range:
#     print(f"Clustering #: {k}")
#     kmeans = clustering(k, data_scaled)
#     df_processed['Cluster'] = kmeans.labels_
#     v_clustering(k, df_processed, pricing_map, get_original_centroids(scaler, kmeans))

# Visualization of methods
v_sse(k_range, data_scaled)

# Clustering with optimal k
kmeans = clustering(optimal_k, data_scaled)

# Add the cluster labels to our processed (but not scaled) DataFrame
df_processed['Cluster'] = kmeans.labels_
v_clustering(optimal_k, df_processed, pricing_map, get_original_centroids(scaler, kmeans))

# Create new datasheet with cluster label
create_new_datasheet(kmeans)
