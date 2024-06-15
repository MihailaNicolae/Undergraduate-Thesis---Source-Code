import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

# Load dataset
dataset = pd.read_csv("GMM_dataset.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
anomaly_indices = np.where(y == -1)[0]
# Find the best number of components
lowest_bic = np.infty
best_gmm = None
best_n_components = 1
for n_components in range(1, 11):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    bic = gmm.bic(X)
    if bic < lowest_bic:
        lowest_bic = bic
        best_gmm = gmm
        best_n_components = n_components

print(f"Best number of components: {best_n_components-1}")

# Fit Gaussian Mixture Model with the best number of components
gmm = best_gmm
gmm.fit(X)

# Predict labels and anomaly scores
y_pred = gmm.predict(X)
anomaly_score = gmm.score_samples(X)

# Set threshold for anomaly detection
threshold = np.percentile(anomaly_score, 4.5)  # e.g., 5th percentile

# Identify anomalies
anomalies = X[anomaly_score < threshold]

#Sorting data for legend
mislabeled_anomalies = X[(anomaly_score > threshold) & (y == -1)]

# Calculate true positive, false positive and false negative
tp = np.sum((y == -1) & (anomaly_score < threshold))
fp = np.sum((y != -1) & (anomaly_score < threshold))
fn = np.sum((y == -1) & (anomaly_score >= threshold))

# Calculate precision, recall, and F1 score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Confusion Matrix:")
print(confusion_matrix(y, anomaly_score < threshold))
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
# Generate grid points for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.contour(xx, yy, Z, levels=np.percentile(Z, [2, 5, 10]), linewidths=2, colors='k', linestyles='dashed')

# Visualize anomalies
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Normal Data (Labeled)')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='green', label='Normal Data (Mislabeled)')
plt.scatter(X[anomaly_indices, 0], X[anomaly_indices, 1], color='red', label='Anomaly (Labeled)')
plt.scatter(mislabeled_anomalies[:, 0], mislabeled_anomalies[:, 1], c='yellow', label='Anomaly (Mislabeled)')
plt.title('Anomaly Detection using Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
