from fastTSNE import TSNE
from sklearn import datasets
import sys
import json 
import numpy as np
import copy

print("Loading tsne data: ", sys.argv[1])
tsne_data = open(sys.argv[1])
data = json.load(tsne_data)
x = data["code"]
y = data["class"]

x2 = data["code2"]
y2 = [x__ + 4 for x__ in data["class2"]]

x_ = copy.deepcopy(x)
y_ = copy.deepcopy(y)
x_.extend(x2)
y_.extend(y2)

x_ = np.array(x_)
y_ = np.array(y_)

print ("x: ", x)
print ("y: ", y)
tsne = TSNE(
    n_components=2, perplexity=20, learning_rate=100, early_exaggeration=10,
    n_jobs=10, initialization='random', metric='euclidean',
    n_iter=10000, early_exaggeration_iter=25, neighbors='exact',
    negative_gradient_method='bh', min_num_intervals=20,
    late_exaggeration_iter=1000, late_exaggeration=5,
)

embedding = tsne.fit(x_)

print ("embedding: ", embedding) 
print ("targets: ", y_) 
x = embedding
import matplotlib.pyplot as plt

unique = list(set(y_))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [x_[j][0] for j  in range(len(x)) if y_[j] == u]
    yi = [x_[j][1] for j  in range(len(x)) if y_[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(u))
plt.legend()

plt.show()