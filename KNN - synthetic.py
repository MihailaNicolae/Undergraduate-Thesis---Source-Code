import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv('KNN_dataset.csv')

# Select features for anomaly detection (adjust as needed)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values  # Assuming the labels are in the first column

# Train the KNN model
k = 24  # You can adjust the number of neighbors as needed
knn_model = NearestNeighbors(n_neighbors=k)
knn_model.fit(X)

# Calculate distances to the k-nearest neighbors for each data point
distances, _ = knn_model.kneighbors(X)

# Calculate the average distance to the k-nearest neighbors for each data point
avg_distances = distances.mean(axis=1)

# Determine a threshold for anomaly detection
# You can adjust this threshold based on your specific requirements
threshold = avg_distances.mean() + 1.4*avg_distances.std()#+ 2 * avg_distances.std()

# Identify anomalies
anomalies = data[avg_distances > threshold]
anomaly_labels = y[avg_distances > threshold]

#Sorting data for legend
mislabeled_anomalies = X[(avg_distances < threshold) & (y == -1)]
mislabeled_normals = X[(avg_distances >= threshold) & (y != -1)]
labeled_anomalies = X[(avg_distances >= threshold) & (y == -1)]

tp = np.sum((y == -1) & (avg_distances >= threshold))
fp = np.sum((y != -1) & (avg_distances >= threshold))
fn = np.sum((y == -1) & (avg_distances < threshold))

# Calculate precision, recall, and F1 score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Confusion Matrix:")
print(confusion_matrix(y, avg_distances < threshold))
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

# Visualize the data and anomalies
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Normal Data (Labeled)')
plt.scatter(mislabeled_normals[:, 0], mislabeled_normals[:, 1], c='green', label='Normal Data (Mislabeled)')
plt.scatter(labeled_anomalies[:, 0], labeled_anomalies[:, 1], c='red', label='Anomaly (Labeled)')
plt.scatter(mislabeled_anomalies[:, 0], mislabeled_anomalies[:, 1], c='yellow', label='Anomaly (Mislabeled)')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Anomaly Detection using KNN')
plt.legend()
plt.show()