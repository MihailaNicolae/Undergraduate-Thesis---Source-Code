import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv("IF_dataset.csv")

# Extract features and labels
X = data.iloc[:, :-1]  # Features
y_true = data.iloc[:, -1]   # True labels

# Fit the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# Predict anomalies
y_pred = model.predict(X)

# Convert 1 (normal) to 0 and -1 (anomaly) to -1
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = -1

#Sorting data for legend
mislabeled_anomalies = [(y_true == -1) & (y_pred == 0)]
mislabeled_normals = [(y_true == 0) & (y_pred == -1)]
labeled_anomalies = [(y_true == -1) & (y_pred == -1)]

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

tp = np.sum((y_true == -1) & (y_pred == -1))
fp = np.sum((y_true == 0) & (y_pred == -1))
fn = np.sum((y_true == -1) & (y_pred == 0))

# Calculate precision, recall, and F1 score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Confusion Matrix:")
print(conf_matrix)
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

# Predict anomaly scores
anomaly_scores = model.decision_function(X)  # The anomaly scores are the negative of the average path length
anomaly_scores = -anomaly_scores  # Convert to positive values for easier interpretation

# Plot outlier scores for each data point
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(anomaly_scores)), anomaly_scores, color='blue', label='Normal Data (Labeled)')
plt.scatter(np.where((y_true == 0) & (y_pred == -1))[0], anomaly_scores[(y_true == 0) & (y_pred == -1)], color='green', label='Normal Data (Mislabeled)')
plt.scatter(np.where((y_true == -1) & (y_pred == -1))[0], anomaly_scores[(y_true == -1) & (y_pred == -1)], color='red', label='Anomaly (Labeled)')
plt.scatter(np.where((y_true == -1) & (y_pred == 0))[0], anomaly_scores[(y_true == -1) & (y_pred == 0)], color='yellow', label='Anomaly (Mislabeled)')
plt.title('Anomaly Detection using Isolation Forest')
plt.xlabel('Data Point Index')
plt.ylabel('Outlier Score')
plt.grid(True)
plt.legend()
plt.show()