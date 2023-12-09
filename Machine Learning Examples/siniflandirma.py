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



#%%

from sklearn.utils import shuffle
df = shuffle(df)

#%%


df_np = df.to_numpy()

X = df_np[:,0:4]
y = df_np[:,4]
#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4)



#%%

from sklearn import svm

model_svm = svm.SVC()

model_svm.fit(X_train, y_train) # train set



#%%
model_svm.predict(X_test) # test set
