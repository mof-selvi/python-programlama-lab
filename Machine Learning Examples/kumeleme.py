#%%

from sklearn import datasets
from sklearn import cluster

import matplotlib.pyplot as plt
import numpy as np


#%%

dataset = datasets.make_circles()[0]

#%%

plt.scatter(dataset[:,0],dataset[:,1])


#%%

algo = cluster.KMeans(n_clusters=2)

clustering = algo.fit(dataset)

#%%

plt.scatter(dataset[:,0], dataset[:,1], c=clustering.labels_)