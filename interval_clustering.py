"""
K-Means clustering of bow shock crossing intervals.
We use PCA for dimensionality reduction.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

check_optimal_params = True
pca_dimensions = 2
drop_features = [
    "Standard Deviation Bx",
    "Standard Deviation By",
    "Standard Deviation Bz",
    "Mean Bx",
    "Mean By",
    "Mean Bz",
    "Median Bx",
    "Median By",
    "Median Bz",
]

# Load interval features dataset
interval_data = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock_crossing_intervals.csv",
    index_col=0,
)

print(interval_data.columns)

if drop_features != [""]:
    interval_data = interval_data.drop(columns=drop_features)

# PCA doesn't work with nan
interval_data.dropna(inplace=True)

# Our features are very different from one another
# We can reduce them all to the same range using a standardised scaler
interval_data[interval_data.columns] = StandardScaler().fit_transform(interval_data)

# K-Means performs poorly with high dimensional data.
# As dimensions increase, data points get closer together in N-D space
# and are hence harder to cluser.
# We reduce the number of dimensions using Principal Component Analysis.
pca = PCA(n_components=pca_dimensions)
pca_result = pca.fit_transform(interval_data)

print(
    "Explained variation per principal component: {}".format(
        pca.explained_variance_ratio_
    )
)

print(
    "Cumulative variance explained by 2 principal components: {:.2%}".format(
        np.sum(pca.explained_variance_ratio_)
    )
)

# Find features imporant to the pca
if pca_dimensions == 3:
    pca_labels = ["PC_1", "PC_2", "PC_3"]
else:
    pca_labels = ["PC_1", "PC_2"]

dataset_pca = pd.DataFrame(
    abs(pca.components_), columns=interval_data.columns, index=pca_labels
)
print("\n\n", dataset_pca)

print("\n*************** Most important features *************************")
for i in range(pca_dimensions):
    print(f"As per PC {i+1}:\n", (dataset_pca[dataset_pca > 0.3].iloc[i]).dropna(), "\n\n")
print("\n******************************************************************")

if check_optimal_params:

    # We need to choose how many clusters we want.
    # We use the silhouette score method to find the optimal number
    parameters = np.arange(2, 20)

    parameter_grid = ParameterGrid({"n_clusters": parameters})
    best_score = -1
    kmeans_model = KMeans()  # instantiating KMeans model
    silhouette_scores = []
    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)  # set current hyper parameter
        kmeans_model.fit(
            pca_result
        )  # fit model on wine dataset, this will find clusters based on parameter p
        ss = metrics.silhouette_score(
            pca_result, kmeans_model.labels_
        )  # calculate silhouette_score
        silhouette_scores += [ss]  # store all the scores
        print("Parameter:", p, "Score", ss)
        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p
    # plotting silhouette score
    plt.bar(
        range(len(silhouette_scores)),
        list(silhouette_scores),
        align="center",
        color="#722f59",
        width=0.5,
    )
    plt.xticks(range(len(silhouette_scores)), list(parameters))
    plt.title("Silhouette Score", fontweight="bold")
    plt.xlabel("Number of Clusters")
    plt.show()

    optimum_num_clusters = parameters[np.argmax(silhouette_scores)]

else:
    optimum_num_clusters = 2

kmeans = KMeans(n_clusters=optimum_num_clusters)
cluster_predictions = kmeans.fit_predict(pca_result)

# Plotting
if pca_dimensions == 3:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

else:
    fig, ax = plt.subplots()


for i in range(optimum_num_clusters):
    if pca_dimensions == 3:
        ax.scatter(
            pca_result[:,0][
                cluster_predictions == i
            ],
            pca_result[:,1][
                cluster_predictions == i
            ],
            pca_result[:,2][
                cluster_predictions == i
            ],
            marker="o"
        )

    else:
        ax.scatter(
            pca_result[:,0][
                cluster_predictions == i
            ],
            pca_result[:,1][
                cluster_predictions == i
            ],
        )

plt.show()
