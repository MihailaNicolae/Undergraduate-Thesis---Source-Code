import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix
from sklearn.mixture import GaussianMixture


# Function to generate synthetic data for anomaly detection
def generate_anomaly_data(n_samples, n_features, n_clusters, anomaly_frac=0.05):
    # Generate data for normal clusters
    X_normal, y_normal = make_gaussian_data(n_samples, n_features, n_clusters)
    # Generate data for anomalies
    n_anomalies = int(anomaly_frac * n_samples)
    X_anomalies = np.random.uniform(low=-10, high=10, size=(n_anomalies, n_features))

    # Combine normal and anomaly data
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([y_normal, np.full(n_anomalies,-1)])

    return X, y


# Function to generate data from Gaussian clusters
def make_gaussian_data(n_samples, n_features, n_clusters):
    covariances = []
    X = []
    y = []

    for cluster in range(n_clusters):
        mean = np.random.randint(-10, 10, size=(n_features,))
        covariance = make_spd_matrix(n_features)  # Generate a random positive definite matrix
        covariances.append(covariance)
        X_cluster = np.random.multivariate_normal(mean, covariance, n_samples // n_clusters)
        X.extend(X_cluster)
        y.extend([cluster] * (n_samples // n_clusters))

    return np.array(X), np.array(y)


# Parameters
n_samples = 1000
n_features = 2
n_clusters = 3
anomaly_frac = 0.05
output_file = "GMM_dataset.csv"

# Generate dataset
X, y = generate_anomaly_data(n_samples, n_features, n_clusters, anomaly_frac)

# Save dataset to CSV
data = np.column_stack((X, y))
columns = [f"feature_{i}" for i in range(n_features)] + ['cluster_label']
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_file, index=False)

print(f"Dataset saved to {output_file}")