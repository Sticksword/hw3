# set plots inline for ipython
# %matplotlib inline

# general python imports
from time import time

# Standard scientific Python imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# import datasets, preprocessing, piplining
from sklearn import datasets, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# dimensionality reduction
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration

from sklearn.neural_network import MLPClassifier, MLPRegressor

# globals
n_row, n_col = 2, 4
n_components = n_row * n_col

# set random seed
rand_state = np.random.RandomState(32)

# function to plot different decompositions of the data
def plot_gallery(title, images, n_col=n_col, n_row=n_row, image_shape = (64, 64)):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    plt.show()

def do_clustering(dataset, clustering_alg_dict, image_shape=(64,64)):
    for title, clustering in clustering_alg_dict.items():
        t0 = time()
        clustering.fit(dataset)
        train_time = (time() - t0)
        if isinstance(clustering, Pipeline):
            clustering = clustering.steps[-1][1]
        if hasattr(clustering, 'means_'):
            components = clustering.means_
        elif hasattr(clustering, 'cluster_centers_'):
            components = clustering.cluster_centers_
        print(type(clustering))
        plot_gallery('%s - Train time %.1fs' % (title, train_time),
                     components[:n_components], image_shape=image_shape)

def do_decompositions(dataset, decomposition_dict, image_shape=(64,64)):
    for title, decomposition in decompositions.items():
        if not title == 'Feature Agglomeration':
            t0 = time()
            decomposition.fit(dataset)
            train_time = (time() - t0)
            plot_gallery('%s - Train time %.1fs' % (title, train_time),
                         decomposition.components_[:n_components], image_shape=image_shape)

# Load datasets
cc_data = np.loadtxt(r'Concrete_Data.csv', delimiter=',', skiprows=1)
cc_y = cc_data[:, 8]
cc_x = cc_data[:, 0:8]

# digit clustering
clustering_algs = {
    'K-Means': KMeans(n_clusters=n_components),
    'Expectation Maximization': GaussianMixture(n_components=n_components)
}

# dimensionality reduction algorithms
decompositions = {
    'Principal Components Analysis': PCA(n_components=n_components, whiten=True),
    'Independent Components Analysis': FastICA(n_components=n_components, whiten=True),
    'Gaussian Random Projections': GaussianRandomProjection(n_components=n_components),
    'Feature Agglomeration': FeatureAgglomeration(n_clusters=32)
}

do_clustering(cc_x, clustering_algs, image_shape=(1,8))

do_decompositions(cc_x, decompositions, image_shape=(1,8))
plt.plot(decompositions['Principal Components Analysis'].explained_variance_)
plt.xlabel('components')
plt.ylabel('explained_variance_')
plt.show()

# # do feature agglomerations
# agglo = FeatureAgglomeration(n_clusters=32, connectivity=grid_to_graph(*digits.images[0].shape))
# agglo.fit(digits_X)
#
# plt.suptitle('Agglomerated Feature Labels', size=16)
#
# plt.subplot(1,2,1)
# plt.imshow(np.reshape(agglo.labels_, digits.images[0].shape),
#            interpolation='nearest', cmap=plt.cm.spectral)
# plt.xticks(())
# plt.yticks(())
#
# # do feature agglomerations
# agglo = FeatureAgglomeration(n_clusters=32, connectivity=grid_to_graph(*faces.images[0].shape))
# agglo.fit(faces_X)
#
# plt.subplot(1,2,2)
# plt.imshow(np.reshape(agglo.labels_, faces.images[0].shape),
#            interpolation='nearest', cmap=plt.cm.spectral)
# plt.xticks(())
# plt.yticks(())
#
# plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)



# split the data for supervised learning portion
X_train, X_test, y_train, y_test = train_test_split(cc_x, cc_y, test_size=0.3, random_state=rand_state)

# n_components = 40

preprocessors = {
    'No preprocess' : None,
    'Principal Components Analysis': PCA(n_components=n_components, whiten=True),
    'Independent Components Analysis': FastICA(n_components=n_components, whiten=True),
    'Gaussian Random Projections': GaussianRandomProjection(n_components=n_components),
    # 'Feature Agglomeration': FeatureAgglomeration(n_clusters=n_components, connectivity=grid_to_graph(*faces.images[0].shape)),
    'K-Means': KMeans(n_clusters=n_components)
}

results = {}
wallclock = {}

for title, process in preprocessors.items():
    if process == None:
        nn_ = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1)
    else:
        nn_ = make_pipeline(process, MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1))
    t0 = time()
    nn_.fit(X_train, y_train)
    wallclock[title] = (time() - t0)
    y_pred = nn_.predict(X_test)
    results[title] = nn_.score(X_test, y_test)

pd.DataFrame.from_dict(results, orient='index').plot(kind='barh', title='Accuracy Scores')
plt.show()
pd.DataFrame.from_dict(wallclock, orient='index').plot(kind='barh', title='Training Times')
plt.show()
