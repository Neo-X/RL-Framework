from fastTSNE import TSNE
from sklearn import datasets
import sys
import json 
import numpy as np

print("Loading tsne data: ", sys.argv[1])
tsne_data = open(sys.argv[1])
data = json.load(tsne_data)
x = np.array(data["code"])
y = np.array(data["class"])
print ("x: ", x)
print ("y: ", y)
tsne = TSNE(
    n_components=2, perplexity=30, learning_rate=100, early_exaggeration=10,
    n_jobs=10, initialization='random', metric='euclidean',
    n_iter=10000, early_exaggeration_iter=25, neighbors='exact',
    negative_gradient_method='bh', min_num_intervals=20,
    late_exaggeration_iter=1000, late_exaggeration=4,
)

embedding = tsne.fit(x)

print ("embedding: ", embedding) 
print ("targets: ", y) 
x = embedding
import matplotlib.pyplot as plt

unique = list(set(y))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [x[j][0] for j  in range(len(x)) if y[j] == u]
    yi = [x[j][1] for j  in range(len(x)) if y[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(u))
plt.legend()

plt.show()