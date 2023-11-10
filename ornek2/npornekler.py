
#%%

import numpy as np


#%%

liste = [1,2,3,4]
liste2 = [4,3,2,1]

np_dizisi = np.array(liste)
np_dizisi2 = np.array(liste2)

# print(liste*5)
# print(np_dizisi*5)
# print("*"*50)

print(np_dizisi * np_dizisi2)


#%%

np.arange(0,10,0.5)

#%%
range(0,10,2)

#%%
for i in range(3,17,4):
    print(i)


#%%


np.arange(0,10,0.5).reshape(5,4)

#%%
np_d2 = np.arange(0,10,0.5)
np_m1 = np_d2.reshape(5,4)

np_m1


#%%

print(np_m1*np_dizisi)

#%%

np_dizisi_t = np_dizisi.reshape(4,1)
print(np_dizisi_t)

#%%

print(np_m1*np_dizisi_t)

#%%

np_dizisi_t.T[0]

#%%


np_m2 = np_m1*np_dizisi

#%%

np_m2[:,0]

#%%

import pandas as pd

#%%

excel_icerik = pd.read_excel("file_example_XLS_50.xls")

excel_icerik