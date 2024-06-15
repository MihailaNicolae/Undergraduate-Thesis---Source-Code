#in PCA_generare_1 generam datele
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Load data from CSV file
data = pd.read_csv("PCA_dataset.csv")

# Separate features and labels
X = data.iloc[:, :-1]  # Selecting all columns except the last one as features
y_true = data.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Calculate reconstruction error
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.mean(np.square(X_scaled - X_reconstructed), axis=1)

# Define a threshold for anomaly detection
threshold = np.percentile(reconstruction_error, 95)  # 95th percentile as threshold 95.3 e cel mai bun

# Predict anomalies
y_pred = np.where(reconstruction_error >= threshold, -1, 0)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Extract true positives, false positives, and false negatives from confusion matrix
tp = np.sum((y_true == -1) & (y_pred == -1))  # Actual anomalies correctly predicted as anomalies
fp = np.sum((y_true == 0) & (y_pred == -1))   # Normal data incorrectly predicted as anomalies
fn = np.sum((y_true == -1) & (y_pred == 0))   # Actual anomalies incorrectly predicted as normal data

# Calculate precision, recall, and F1-score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

# Visualize results
plt.figure(figsize=(8, 6))

#Sorting data for legend
mis_anomalies = X_pca[(y_pred == 0) & (y_true == -1)]
anomalies = X_pca[(y_pred == -1) & (y_true == -1)]
mis_normals = X_pca[(y_pred == -1) & (y_true == 0)]

# Color the points according to their label
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='blue', label='Normal Data (Labeled)')
plt.scatter(mis_normals[:, 0], mis_normals[:, 1], color='green', label='Normal Data (Mislabeled)')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red', label='Anomalies (Labeled)')
plt.scatter(mis_anomalies[:, 0], mis_anomalies[:, 1], color='yellow', label='Anomalies (Mislabeled)')

plt.title('Anomaly Detection using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.grid(True)
plt.show()