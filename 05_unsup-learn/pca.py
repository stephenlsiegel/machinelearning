# PCA in sklearn (this won't run, just for demonstration)

from sklearn.decomposition import PCA

# initialize PCA object
pca = PCA(n_components=2)

# fit PCA to data
pca.fit(data)

pca.explained_variance_ratio_ # percentage of variance explained by each of the selected components

first_pc = pca.components_[0] # first principal component
second_pc = pca.components_[1]

# transform data
transformed_data = pca.transform(data)