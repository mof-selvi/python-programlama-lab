#%%

from sklearn import datasets
from sklearn import cluster

import matplotlib.pyplot as plt
import numpy as np


iris = datasets.load_iris()

#%%
# my_ds = []

# idx=0
# for i in iris["data"]:
#     data = i+iris["target"][idx]
#     idx+=1

#     my_ds.append(data)

#%%

import pandas as pd


df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])

df["target"] = iris["target"]

