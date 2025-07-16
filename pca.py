import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Load your CSV file
df = pd.read_csv("less_equal_5.csv")  # Replace with your actual file path

# Select first 6 PID columns
features = df.columns[:6]
X = df[features].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Print explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Plot variance explained
plt.figure(figsize=(5, 4))
plt.bar(['PC1', 'PC2','PC3'], pca.explained_variance_ratio_ * 100)
plt.ylabel('Variance Explained (%)')
plt.title('PCA Explained Variance')
plt.tight_layout()
plt.show()

# Plot heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of PID Parameters')
plt.tight_layout()
plt.show()

# Elbow method for optimal k
inertia = []
K = range(1, min(len(X_pca), 10))  # max clusters = number of samples
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(6, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.tight_layout()
plt.show()

# Choose number of clusters (set manually or from elbow plot)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_pca)
centroids = kmeans.cluster_centers_

# Plot PCA + clusters
# Plot PCA + clusters with cluster labels in legend
plt.figure(figsize=(8, 6))

# Plot each cluster separately with its own label
for cluster_id in range(optimal_k):
    cluster_points = X_pca[clusters == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                s=100, edgecolor='k', label=f'Cluster {cluster_id}')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1],
            c='red', marker='X', s=250, label='Centroids')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA + KMeans Clustering')
plt.legend(title='Legend')
plt.tight_layout()
plt.show()

from sklearn.metrics import silhouette_samples, silhouette_score

# Calculate silhouette scores
silhouette_vals = silhouette_samples(X_pca, clusters)
avg_silhouette = silhouette_score(X_pca, clusters)
df['Cluster'] = clusters
df['Silhouette'] = silhouette_vals

# Average silhouette score per cluster
cluster_silhouette = df.groupby('Cluster')['Silhouette'].mean().sort_values(ascending=False)

print("\nAverage Silhouette Score (Overall):", round(avg_silhouette, 4))
print("\nAverage Silhouette Score per Cluster:")
print(cluster_silhouette)

# Identify best cluster (highest avg silhouette score)
best_cluster = cluster_silhouette.idxmax()
print(f"\n✅ Best-performing cluster is Cluster {best_cluster} (highest cohesion)")

# Inverse transform PCA centroids back to original feature space
centroids_original = scaler.inverse_transform(pca.inverse_transform(centroids))
centroids_original_df = pd.DataFrame(centroids_original, columns=features)
centroids_original_df.index.name = 'Cluster'

print("\nCluster Centroids in Original PID Parameter Space:")
print(centroids_original_df.round(4))
