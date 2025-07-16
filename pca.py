import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Load your CSV file
df = pd.read_csv("greater_than_5.csv")  # Replace with your actual file path

# Select first 6 PID columns
features = df.columns[:8]
X = df[features].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Print explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Plot variance explained
plt.figure(figsize=(5, 4))
plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_ * 100)
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
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_pca)
centroids = kmeans.cluster_centers_

# Plot PCA + clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=100, edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA + KMeans Clustering')
plt.legend()
plt.tight_layout()
plt.show()

# Print cluster centroids
centroids_df = pd.DataFrame(centroids, columns=['PC1', 'PC2'])
centroids_df.index.name = 'Cluster'
print("\nCluster Centroids (in PCA space):")
print(centroids_df)
