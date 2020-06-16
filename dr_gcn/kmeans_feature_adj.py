import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

## loading the npy file, surf descriptor for every image
surf_des = next(os.walk('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/aptos/des_brisk'))[2]

surf_des.sort()
print(len(surf_des))

whole_surf = np.empty([0, 128])
counter = 0
length = []

## load the files for clustering
for descriptor in surf_des:
    test = np.load('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/aptos/des_brisk/{}'.format(descriptor))
    length.append(test.shape[0])
    whole_surf = np.append(whole_surf, test, 0)

## define the function to find correlation in one image
def all_np(arr):
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result

## show the cloustering info
n_classes = 20
n_samples = whole_surf.shape[0]
n_features = whole_surf.shape[1]
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_classes, n_samples, n_features))

## do the clustering
cluster = KMeans(n_clusters=20, init='k-means++', n_init=10)
cluster.fit(whole_surf)

# TODO
# 有必要normalize吗
## normalize feature vector and save as  feature_vector.pkl
minmax_normalizer = MinMaxScaler()
gaussian_normalizer = StandardScaler()
'''using different normalization method'''
feature_vector = minmax_normalizer.fit_transform(cluster.cluster_centers_)
# feature_vector = gaussian_normalizer.fit_transform(cluster.cluster_centers_)

feature_tosave = open('descripion/feature_vector_brisk.pkl', 'wb')
pickle.dump(feature_vector, feature_tosave, -1)
feature_tosave.close()
print('feature_vector shape ',feature_vector.shape)

## define a dictionary structure for num and adj correlation and save pkl file
dr_adj = {}
# assume they are random distributedq1
'''使用原先的先定义数字'''
dr_adj['nums'] = np.array([238,  243,    330,  181,  244,  186,  713,  337,  445,  141,  200,  421,  287,  245, 2008,  245,  96,  229,  261,  256])
# dr_adj['nums'] = np.arange(20)
adj_matrix = np.zeros([20, 20])
start_point = 0
end_point = 0
counter = 0
for len in length:
    start_point = end_point
    end_point = len + start_point
    temp = cluster.labels_[start_point:end_point]
    dicti = all_np(temp)

    add_vec = np.zeros([20])
    for key in dicti.keys():
        add_vec[key.item()] = dicti[key]

    for key in dicti.keys():
        adj_matrix[key.item()] = adj_matrix[key.item()] + add_vec
        # erase itself value
        adj_matrix[key.item()][key.item()] -= dicti[key]
'''使用occurence次数 论文提出'''
# dr_adj['nums'] = np.sum(adj_matrix, axis=0)

dr_adj['adj'] = adj_matrix
print('show dr_adj, keys:{}\t value: dr_adj[nums]:{} \t dr_adj[adj]:{}\n'
      .format(dr_adj.keys(), dr_adj['nums'].shape, dr_adj['adj'].shape))
## save adj pkl file
adj_tosave = open('descripion/dr_adj_brisk.pkl', 'wb')
pickle.dump(dr_adj, adj_tosave, -1)
adj_tosave.close()
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(whole_surf)
kmeans = KMeans(init='k-means++', n_clusters=n_classes, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
plt.savefig('check.png')