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

# Load interval features dataset
interval_data = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock_crossing_intervals.csv",
    index_col=0,
)

# PCA doesn't work with nan
interval_data.dropna(inplace=True)

# Our features are very different from one another
# We can reduce them all to the same range using a standardised scaler
interval_data[interval_data.columns] = StandardScaler().fit_transform(interval_data)

# K-Means performs poorly with high dimensional data.
# As dimensions increase, data points get closer together in N-D space
# and are hence harder to cluser.
# We reduce the number of dimensions using Principal Component Analysis.
pca = PCA(n_components=2)
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
dataset_pca = pd.DataFrame(
    abs(pca.components_), columns=interval_data.columns, index=["PC_1", "PC_2"]
)
print("\n\n", dataset_pca)

print("\n*************** Most important features *************************")
print("As per PC 1:\n", (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())
print("\n\nAs per PC 2:\n", (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())
print("\n******************************************************************")

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
        interval_data
    )  # fit model on wine dataset, this will find clusters based on parameter p
    ss = metrics.silhouette_score(
        interval_data, kmeans_model.labels_
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

kmeans = KMeans(n_clusters=optimum_num_clusters)
kmeans.fit(interval_data)

centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)

plt.show()
