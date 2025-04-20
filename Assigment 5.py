import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Function to load dataset
def load_dataset(dataset_name):
    if dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError("Dataset not supported. Choose 'wine', 'iris', or 'breast_cancer'")
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

# Load a dataset
X, y = load_dataset('wine')  # You can change to 'iris' or 'breast_cancer'
print(f"Dataset shape: {X.shape}")

# Define preprocessing functions
def no_preprocessing(X):
    return X

def apply_normalization(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def apply_transformation(X):
    # Apply log transformation (add small constant to avoid log(0))
    return np.log1p(X)

def apply_pca(X, n_components=0.95):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def apply_transform_and_normalize(X):
    # Apply log transformation followed by normalization
    X_transformed = np.log1p(X)
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_transformed)

def apply_transform_normalize_pca(X, n_components=0.95):
    # Apply log transformation, normalization, and PCA
    X_transformed = np.log1p(X)
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_transformed)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_normalized)

# Define clustering algorithms
def apply_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels

def apply_hierarchical(X, n_clusters):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(X)
    return labels

def apply_meanshift(X, n_clusters=None):  # n_clusters is ignored for MeanShift
    # Estimate bandwidth
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    if bandwidth == 0:
        bandwidth = 0.5  # Fallback if bandwidth estimation fails
    
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(X)
    
    # Print the actual number of clusters found
    n_clusters_found = len(np.unique(labels))
    print(f"Mean Shift found {n_clusters_found} clusters")
    
    return labels

# Define evaluation metrics
def evaluate_clustering(X, labels):
    if len(np.unique(labels)) < 2:
        return "NA", "NA", "NA"  # Cannot evaluate with less than 2 clusters
    
    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = "NA"
    
    try:
        calinski = calinski_harabasz_score(X, labels)
    except:
        calinski = "NA"
    
    try:
        davies = davies_bouldin_score(X, labels)
    except:
        davies = "NA"
    
    return silhouette, calinski, davies

# Define preprocessing techniques
preprocessing_techniques = {
    "No Data Processing": no_preprocessing,
    "Using Normalization": apply_normalization,
    "Using Transform": apply_transformation,
    "Using PCA": apply_pca,
    "Using T+N": apply_transform_and_normalize,
    "T+N+PCA": apply_transform_normalize_pca
}

# Define clustering algorithms
clustering_algorithms = {
    "K-Mean Clustering": apply_kmeans,
    "Hierarchical Clustering": apply_hierarchical,
    "K-mean Shift Clustering": apply_meanshift
}

# Define number of clusters to try
cluster_range = [3, 4, 5]

# Create results structure
results = {}
for alg_name in clustering_algorithms.keys():
    results[alg_name] = {}
    for prep_name in preprocessing_techniques.keys():
        results[alg_name][prep_name] = {
            "Silhouette": [],
            "Calinski-Harabasz": [],
            "Davies-Bouldin": []
        }

# Perform clustering with all combinations
for alg_name, alg_func in clustering_algorithms.items():
    print(f"\nRunning {alg_name}...")
    
    for prep_name, prep_func in preprocessing_techniques.items():
        print(f"  With {prep_name}...")
        
        try:
            # Apply preprocessing
            X_processed = prep_func(X)
            
            # Run clustering for different numbers of clusters
            for n_clusters in cluster_range:
                if alg_name == "K-mean Shift Clustering":
                    # MeanShift doesn't take n_clusters parameter
                    labels = alg_func(X_processed)
                else:
                    labels = alg_func(X_processed, n_clusters)
                
                # Evaluate clustering
                silhouette, calinski, davies = evaluate_clustering(X_processed, labels)
                
                # Store results
                results[alg_name][prep_name]["Silhouette"].append(silhouette)
                results[alg_name][prep_name]["Calinski-Harabasz"].append(calinski)
                results[alg_name][prep_name]["Davies-Bouldin"].append(davies)
                
                print(f"    clusters={n_clusters}, silhouette={silhouette}, calinski={calinski}, davies={davies}")
        
        except Exception as e:
            print(f"    Error with {prep_name}: {e}")
            # Fill with NA values if error occurs
            for n_clusters in cluster_range:
                results[alg_name][prep_name]["Silhouette"].append("NA")
                results[alg_name][prep_name]["Calinski-Harabasz"].append("NA")
                results[alg_name][prep_name]["Davies-Bouldin"].append("NA")

# Function to format values properly
def format_value(val, metric):
    if val == "NA":
        return "NA"
    elif not isinstance(val, (int, float)):
        return "NA"
    elif metric == "Silhouette":
        return f"{val:.2f}"
    elif metric == "Calinski-Harabasz":
        return f"{int(val)}"
    else:  # Davies-Bouldin
        return f"{val:.2f}"

def create_formatted_table(alg_name):
    df = pd.DataFrame(columns=[
        "Algorithm",
        "Preprocessing",
        "Clusters",
        "Silhouette Score",
        "Calinski-Harabasz Score",
        "Davies-Bouldin Score"
    ])

    for prep_name in results[alg_name]:
        for k in results[alg_name][prep_name]:
            metrics = results[alg_name][prep_name][k]
            row_data = {
                "Algorithm": alg_name,
                "Preprocessing": prep_name,
                "Clusters": k,
                "Silhouette Score": metrics[0],
                "Calinski-Harabasz Score": metrics[1],
                "Davies-Bouldin Score": metrics[2]
            }
            df.loc[len(df)] = row_data

    return df



# Generate and display tables for each algorithm
for alg_name in clustering_algorithms.keys():
    print(f"\n\n{'-'*80}")
    print(f"Using {alg_name}")
    print(f"{'-'*80}")
    
    # Generate the table
    table_df = create_formatted_table(alg_name)
    
    # Output a well-formatted version
    print(table_df.to_string(index=False))
    
    # Save to CSV
    table_df.to_csv(f"{alg_name.replace(' ', '_')}_results.csv", index=False)

# Create the table exactly like the image - with preprocessing techniques as column groups
def create_image_like_table(alg_name):
    # Create the header for the table
    print(f"\n{'='*120}")
    print(f"Using {alg_name}")
    print(f"{'='*120}")
    
    # Create a separator line for the table
    print("-" * 120)
    
    # Print the column headers for parameters and preprocessing techniques
    header_format = "{:<15} | "
    for prep_name in preprocessing_techniques.keys():
        header_format += "{:^17} | "
    
    # Print the preprocessing techniques as column groups
    print(header_format.format("", *preprocessing_techniques.keys()))
    
    # Print a separator line
    print("-" * 120)
    
    # Print the cluster numbers
    cluster_header = "{:<15} | "
    for _ in preprocessing_techniques.keys():
        for c in cluster_range:
            cluster_header += "{:^5} | "
    
    cluster_values = ["Parameters"]
    for _ in preprocessing_techniques.keys():
        for c in cluster_range:
            cluster_values.append(f"c={c}")
    
    print(cluster_header.format(*cluster_values))
    
    # Print a separator line
    print("-" * 120)
    
    # Print rows for each metric
    for metric in ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]:
        row_format = "{:<15} | "
        for _ in preprocessing_techniques.keys():
            for _ in cluster_range:
                row_format += "{:^5} | "
        
        row_values = [metric]
        for prep_name in preprocessing_techniques.keys():
            for i, c in enumerate(cluster_range):
                val = results[alg_name][prep_name][metric][i]
                row_values.append(format_value(val, metric))
        
        print(row_format.format(*row_values))
        
        # Print a separator line after each metric
        print("-" * 120)

# Generate and display tables in the format of the image
for alg_name in clustering_algorithms.keys():
    create_image_like_table(alg_name)

# Export table data in a clean format for a report
for alg_name in clustering_algorithms.keys():
    print(f"\nExporting data for {alg_name}...")
    
    # Create a multi-index dataframe
    cols = pd.MultiIndex.from_product([
        preprocessing_techniques.keys(),
        [f"c={c}" for c in cluster_range]
    ])
    
    # Create dataframe with metrics as rows
    df = pd.DataFrame(index=["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"], columns=cols)
    
    # Fill the dataframe
    for metric in ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]:
        for prep_name in preprocessing_techniques.keys():
            for i, c in enumerate(cluster_range):
                val = results[alg_name][prep_name][metric][i]
                df.loc[metric, (prep_name, f"c={c}")] = format_value(val, metric)
    
    # Export to CSV
    df.to_csv(f"{alg_name.replace(' ', '_')}_formatted_results.csv")

# Find the best performing configuration
def find_best_configuration():
    best_score = -float('inf')
    best_config = None
    
    for alg_name in clustering_algorithms.keys():
        for prep_name in preprocessing_techniques.keys():
            for i, c in enumerate(cluster_range):
                val = results[alg_name][prep_name]["Silhouette"][i]
                if val != "NA" and isinstance(val, (int, float)) and val > best_score:
                    best_score = val
                    best_config = (alg_name, prep_name, c)
    
    return best_config, best_score

# Display the best configuration
best_config, best_score = find_best_configuration()
print("\n\nBEST CONFIGURATION:")
print(f"Algorithm: {best_config[0]}")
print(f"Preprocessing: {best_config[1]}")
print(f"Number of clusters: {best_config[2]}")
print(f"Silhouette score: {best_score:.4f}")

# Visualize the best configuration
def visualize_best_clustering(algorithm, preprocessing, clusters, _silhouette):
    print(f"Visualizing the best clustering configuration...")

    # Load and preprocess data accordingly
    X = load_wine().data

    if preprocessing == "Using PCA":
        X_processed = PCA(n_components=2).fit_transform(X)
    elif preprocessing == "Using Transform":
        X_processed = np.log1p(X)
    elif preprocessing == "Using Normalization":
        X_processed = normalize(X)
        X_processed = StandardScaler().fit_transform(X_processed)
    elif preprocessing == "T+N":
        X_processed = np.log1p(X)
        X_processed = normalize(X_processed)
    elif preprocessing == "T+N+PCA":
        X_processed = np.log1p(X)
        X_processed = normalize(X_processed)
        X_processed = PCA(n_components=2).fit_transform(X_processed)
    else:  # No processing
        X_processed = X

    # Perform clustering again using the best config
    if algorithm == "K-Mean Clustering":
        clusterer = KMeans(n_clusters=int(clusters), random_state=42)
    elif algorithm == "Hierarchical Clustering":
        clusterer = AgglomerativeClustering(n_clusters=int(clusters))
    else:
        clusterer = MeanShift()  # Should be 1 cluster number, but MeanShift finds it itself

    labels = clusterer.fit_predict(X_processed)

    # If PCA wasn't used, reduce to 2D for plotting
    if preprocessing not in ["Using PCA", "T+N+PCA"]:
        X_2d = PCA(n_components=2).fit_transform(X_processed)
    else:
        X_2d = X_processed

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(f"{algorithm} ({preprocessing}) - {clusters} Clusters")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()


# Visualize the best configuration
if best_config:
    visualize_best_clustering(*best_config, best_score)


# Generate a summary report
print("\n\n" + "="*80)
print("CLUSTERING ANALYSIS SUMMARY REPORT")
print("="*80)

print("\nDATASET:")
print(f"Dataset name: {load_wine().DESCR.split('===')[0].strip()}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")

print("\nBEST CONFIGURATION:")
print(f"Algorithm: {best_config[0]}")
print(f"Preprocessing technique: {best_config[1]}")
print(f"Number of clusters: {best_config[2]}")
print(f"Silhouette score: {best_score:.4f}")

print("\nKEY FINDINGS:")
# Find best preprocessing across all algorithms
prep_scores = {}
for prep_name in preprocessing_techniques.keys():
    scores = []
    for alg_name in clustering_algorithms.keys():
        for i in range(len(cluster_range)):
            val = results[alg_name][prep_name]["Silhouette"][i]
            if val != "NA" and isinstance(val, (int, float)):
                scores.append(val)
    if scores:
        prep_scores[prep_name] = np.mean(scores)

best_prep = max(prep_scores.items(), key=lambda x: x[1])
print(f"1. Best preprocessing technique: {best_prep[0]} (Avg Silhouette: {best_prep[1]:.4f})")

# Find best algorithm across all preprocessings
alg_scores = {}
for alg_name in clustering_algorithms.keys():
    scores = []
    for prep_name in preprocessing_techniques.keys():
        for i in range(len(cluster_range)):
            val = results[alg_name][prep_name]["Silhouette"][i]
            if val != "NA" and isinstance(val, (int, float)):
                scores.append(val)
    if scores:
        alg_scores[alg_name] = np.mean(scores)

best_alg = max(alg_scores.items(), key=lambda x: x[1])
print(f"2. Best clustering algorithm: {best_alg[0]} (Avg Silhouette: {best_alg[1]:.4f})")

# Find optimal cluster count
cluster_scores = {}
for c_idx, c in enumerate(cluster_range):
    scores = []
    for alg_name in clustering_algorithms.keys():
        for prep_name in preprocessing_techniques.keys():
            val = results[alg_name][prep_name]["Silhouette"][c_idx]
            if val != "NA" and isinstance(val, (int, float)):
                scores.append(val)
    if scores:
        cluster_scores[c] = np.mean(scores)

best_cluster = max(cluster_scores.items(), key=lambda x: x[1])
print(f"3. Optimal number of clusters: {best_cluster[0]} (Avg Silhouette: {best_cluster[1]:.4f})")

print("\nCONCLUSIONS:")
print(f"Based on this analysis, we recommend using {best_config[0]} with {best_config[1]} and {best_config[2]} clusters for this dataset.")
print(f"This combination provides the best clustering performance as measured by the Silhouette score ({best_score:.4f}).")
print("Transform+Normalization and PCA techniques should be carefully considered as they can significantly impact clustering performance.")