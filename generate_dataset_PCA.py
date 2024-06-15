import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

def generate_dataset(n_features, n_samples, n_anomalies):
    # Generate a dataset with two informative features
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=2, n_redundant=8,
                               n_clusters_per_class=1, random_state=42, flip_y=0, class_sep=5.0, n_classes=1)

    # Add anomalies
    anomalies = np.random.uniform(low=-10, high=10, size=(n_anomalies, n_features))
    X = np.concatenate([X, anomalies])
    y = np.concatenate([y, [-1] * n_anomalies])

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Check explained variance
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Explained variance by the first two principal components: {explained_variance * 100:.2f}%")

    #return X_pca, y
    return X,y

X, y = generate_dataset(10, 1000, 50)

# Save to CSV
np.savetxt('PCA_dataset.csv', np.hstack((X, y.reshape(-1,1))), delimiter=',', fmt='%f')