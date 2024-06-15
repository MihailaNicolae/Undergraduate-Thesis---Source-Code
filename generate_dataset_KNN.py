import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

def generate_dataset(num_clusters, num_anomalies):
    # Generating a synthetic dataset with specified number of clusters
    X, _ = make_blobs(n_samples=1000, centers=num_clusters, n_features=2, cluster_std=1, random_state=42)

    # Introducing anomalies
    anomalies = np.random.uniform(low=-10, high=10, size=(num_anomalies, 2))
    X = np.concatenate([X, anomalies])

    # Generating labels
    y = np.zeros(len(X))
    y[-num_anomalies:] = -1  # Labels anomalies as 1

    # Creating a DataFrame
    df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    df['Label'] = y

    return df

# Get user input for number of clusters
num_clusters = int(input("Enter the number of clusters you want in the dataset: "))

# Get user input for number of anomalies
num_anomalies = int(input("Enter the number of anomalies you want in the dataset: "))

# Generate dataset
dataset = generate_dataset(num_clusters, num_anomalies)

# Save the dataset to a CSV file
dataset.to_csv('KNN_dataset.csv', index=False)

print("Dataset saved successfully.")