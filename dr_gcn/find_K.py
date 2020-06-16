import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import xlrd

## loading the npy file, surf descriptor for every image
surf_des = next(os.walk('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/'
                        'Aiki_data/DataSet/Dataset2019/surf_descriptor'))[2]

surf_des.sort()
print(len(surf_des))

whole_surf = np.empty([0, 64])
counter = 0
length = []

## load the files for clustering
for descriptor in surf_des:
    test = np.load('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/Dataset2019/surf_descriptor/{}'.format(descriptor))
    length.append(test.shape[0])
    whole_surf = np.append(whole_surf, test, 0)


featureList = ['Age', 'Gender', 'Degree']
mdl = pd.DataFrame.from_records(whole_surf)

# '利用SSE选择k'
'''选择最佳的k值'''
SSE = []  # 存放每次结果的误差平方和
for k in range(1, 21):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(np.array(mdl))
    SSE.append(estimator.inertia_)
X = range(1, 21)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()

# ## do the clustering
# cluster = KMeans(n_clusters=20, init='k-means++', n_init=10)
# cluster.fit(whole_surf)

