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
import matplotlib.patches as patches
from matplotlib.path import Path

fig, ax = plt.subplots()

unique = list(set(y_))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
encodings = []
for i, u in enumerate(unique):
    print("plotting: ", u)
    xi = [x_[j][0] for j  in range(len(x)) if y_[j] == u]
    yi = [x_[j][1] for j  in range(len(x)) if y_[j] == u]
    print ("xi: ", xi)
    ax.scatter(xi, yi, c=colors[i], label=str(u))
    # ax.plot(xi, yi, c=colors[i], label=str(u))
    encodings.append([xi, yi, u])

print ("encodings[0][0]: ", encodings[0][0])

for i in range(len(encodings[0][0])):
    point0 = [encodings[0][0][i], encodings[0][1][i]]
    point1 = [encodings[1][0][i], encodings[1][1][i]]
    line_ = [[point0[0], point1[0]], [point0[1], point1[1]]]
    print ("line_: ", line_)
    # verts = [(x1,y1), (x2,y2)]
    verts = line_
    codes = [Path.MOVETO,Path.LINETO]
    path = Path(verts, codes)
    ax.add_patch(patches.PathPatch(path, color='green', lw=0.5))
    # plt.plot(line_, 
    #           color = 'black')

# plt.legend()
plt.show()
