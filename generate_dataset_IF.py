import numpy as np
import pandas as pd
import random

def generate_dependency_structure(num_features):
    dependency_structure = []
    selectable_dependencies = []
    for i in range(num_features):
        dependent = np.random.choice([0, -1], p=[0.8, 0.2])  # 80% chance of being independent
        if dependent == 0:
            dependency_structure.append([0])
            selectable_dependencies.append(i)
        else:
            dependency_structure.append([-1])
    for i in range(num_features):
        if dependency_structure[i][0] == -1:
            num_dependent_on = np.random.randint(2, 4)  # Random number of features to depend on (1 to 3)
            dependency_structure[i].extend(random.sample(selectable_dependencies,num_dependent_on))
    return dependency_structure




def generate_data(num_features, num_normal, num_anomalies, dependency_structure):
    # Set means and standard deviations for normal distributions
    feature_means = np.random.uniform(-5, 5, num_features)  # Random means
    feature_stddevs = np.random.uniform(0.5, 2, num_features)  # Random standard deviations

    normal_data = np.zeros((num_normal, num_features))
    for i in range(num_features):
        if dependency_structure[i][0] == 0:
            normal_data[:, i] = np.random.normal(feature_means[i], feature_stddevs[i], num_normal)

    for k in range(len(normal_data)):
        for i in range(num_features):
            if dependency_structure[i][0] == -1:
                suma = 0
                for j in dependency_structure[i][1:]:
                    suma = suma + normal_data[k][j]
                normal_data[k][i] = suma/(len(dependency_structure[i]) - 1)
    # Generate anomalies
    anomalies = np.zeros((num_anomalies, num_features))
    for i in range(num_features):
        if dependency_structure[i][0] == 0:
            anomalies[:, i] = np.random.normal(feature_means[i], feature_stddevs[i], num_anomalies)

    for k in range(len(anomalies)):
        for i in range(num_features):
            if dependency_structure[i][0] == -1:
                suma = 0
                for j in dependency_structure[i][1:]:
                    suma = suma + anomalies[k][j]
                anomalies[k][i] = suma / (len(dependency_structure[i]) - 1)

    # Defining the anomalous features
    for k in range(len(anomalies)):
        for i in range(num_features):
            probability = random.random() * 100
            if probability <= 10 and dependency_structure[i][0] == 0:
                anomalies[k][i] = anomalies[k][i] * 10
            elif probability <= 10 and dependency_structure[i][0] == -1:
                anomalies[k][i] = random.random() * 100

    # Combine normal data and anomalies
    data = np.vstack((normal_data, anomalies))

    # Label the data
    labels = np.zeros(num_normal + num_anomalies)  # 0 for normal data
    labels[num_normal:] = -1  # -1 for anomalies

    # Combine data with labels
    labeled_data = np.column_stack((data, labels))

    return labeled_data


def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, header=False, index=False)


def generate_and_save_dataset(num_features, num_normal, num_anomalies, filename):
    dependency_structure = generate_dependency_structure(num_features)
    with open("dependency_structure.txt", "w") as file:
        file.write(str(dependency_structure))
    data = generate_data(num_features, num_normal, num_anomalies, dependency_structure)
    save_to_csv(data, filename)


# Set parameters
num_features = 100  # Number of features
num_normal = 1000  # Number of normal data points
num_anomalies = 50  # Number of anomalies
filename = "IF_dataset.csv"  # Output filename

# Generate and save dataset
generate_and_save_dataset(num_features, num_normal, num_anomalies, filename)