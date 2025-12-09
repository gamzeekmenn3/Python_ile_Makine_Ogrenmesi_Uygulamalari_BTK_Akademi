from sklearn.datasets import make_classification, make_moons, make_circles
import matplotlib.pyplot as plt
import numpy as np

X, y = make_classification(n_features = 2, n_redundant = 0, n_informative = 2, n_clusters_per_class = 1,random_state = 42)
X += 1.2 * np.random.uniform(size = X.shape)
Xy = (X, y)
# plt.scatter(X[:, 0], X[:, 1], c = y)

# X, y = make_moons(noise = 0.2, random_state = 42)
# plt.scatter(X[:, 0], X[:, 1], c = y)

# X, y = make_circles(noise = 0.1, factor = 0.3, random_state = 42)
# plt.scatter(X[:, 0], X[:, 1], c = y) 

datasets = [Xy,
            make_moons(noise = 0.2, random_state = 42),
            make_circles(noise = 0.1, factor = 0.3, random_state = 42)]    

fig = plt.figure(figsize = (6, 9))
i = 1
for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    
    ax = plt.subplot(len(datasets), 1, i)
    ax.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.coolwarm, edgecolors = "black")
    i += 1
plt.show()
