import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Load and clean data
df = pd.read_csv('combined_with_headers.csv')
df = df[df['Kp_theta'] != 'Kp_theta']
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Define columns
pid_cols = ['Kp_theta', 'Ki_theta', 'Kd_theta', 'Kp_pos', 'Ki_pos', 'Kd_pos']
outcome_cols = ['Avg_Time_s', 'Perfect_Trials']
all_cols = pid_cols + outcome_cols

# Standardize
scaler = StandardScaler()
X_all = scaler.fit_transform(df[all_cols])

### === AUTOMATIC PCA COMPONENT SELECTION (≥ 95% VARIANCE) === ###
pca_full = PCA()
pca_full.fit(X_all)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)
target_var = 0.85
n_components = np.argmax(cum_var >= target_var) + 1
print(f"Auto-selected {n_components} PCA components (≥ {target_var*100:.1f}% variance)")

# Fit PCA with selected components
pca_opt = PCA(n_components=n_components)
X_all_pca = pca_opt.fit_transform(X_all)

# Add first 3 components (if available) for plotting
df['PC1'] = X_all_pca[:, 0]
df['PC2'] = X_all_pca[:, 1]
if n_components >= 3:
    df['PC3'] = X_all_pca[:, 2]

# Plot cumulative variance
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
plt.axhline(y=target_var, color='red', linestyle='--', label=f'{target_var*100:.0f}% variance')
plt.axvline(x=n_components, color='green', linestyle='--', label=f'n = {n_components}')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of PCA Components")
plt.ylabel("Cumulative Variance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

### === ELBOW METHOD FOR CLUSTERING === ###
def elbow_plot(X, title):
    inertias = []
    for k in range(1, 15):
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 15), inertias, 'o-')
    plt.title(f'Elbow Method: {title}')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

elbow_plot(X_all_pca, 'ALL PCA Data')

### === CLUSTERING (manually adjust k after elbow plot) === ###
k = 5  # or choose based on elbow
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_all_pca)

# Silhouette score
sil = silhouette_score(X_all_pca, df['Cluster'])
print(f"Silhouette Score (k={k}): {sil:.3f}")

### === 2D PCA SCATTER === ###
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=80)
plt.title('2D PCA with Clusters')
plt.grid(True)
plt.tight_layout()
plt.show()

# ### === 3D PCA SCATTER (if possible) === ###
# if n_components >= 3:
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(df['PC1'], df['PC2'], df['PC3'], c=df['Cluster'], cmap='Set2', s=60, alpha=0.8)
#     ax.set_title("3D PCA - All Features")
#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")
#     ax.set_zlabel("PC3")
#     plt.tight_layout()
#     plt.show()

### === CLUSTER HEATMAP === ###
cluster_means = df.groupby('Cluster')[all_cols].mean()
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title("Cluster-wise Mean PID + Outcome Values")
plt.tight_layout()
plt.show()

### === BEST CLUSTERS & PARAM SELECTION === ###
sorted_clusters = cluster_means.sort_values(by=['Avg_Reward', 'Perfect_Trials'], ascending=False)
best_clusters = sorted_clusters.head(2).index.tolist()
print(f"\nTop Clusters Based on Reward + Success: {best_clusters}")

# Print top 5 from each best cluster
for cluster in best_clusters:
    print(f"\n--- Top 5 PIDs from Cluster {cluster} ---")
    best = df[df['Cluster'] == cluster].sort_values(by=['Avg_Reward', 'Perfect_Trials'], ascending=False).head(30)
    print(best[pid_cols + outcome_cols])
