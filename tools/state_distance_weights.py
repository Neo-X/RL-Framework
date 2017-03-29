import numpy as np
# import scipy
from scipy.spatial.distance import pdist, squareform
from theano import tensor as T

s = np.array([[ 0.4,  0.7],
              [-0.2, -0.3],
              [ 0.0, -0.4],
              [-0.7,  0.3]])

a = np.array([[ 0.1,  0.2],
              [ 0.2,  0.7],
              [-0.3,  0.1],
              [ 0.0, -0.4]])

s_squared = s**2
print ("s^2 ", s_squared)

# s_dist = s_squared - np.transpose(s_squared)
dist = pdist(s, metric='euclidean')
dist = squareform(dist)
print ("Dist: ", dist)

# X is an m-by-n matrix (rows are examples, columns are dimensions)
# D is an m-by-m symmetric matrix of pairwise Euclidean distances
a = np.sum(s**2, axis=1)
D = np.sqrt((a + a[np.newaxis].T) - 2*np.dot(s, s.T))
print("D shape: ", D.shape)
print( "Dist2: ", D)

avgD = np.mean(D, axis=1)
print ("Avg distance: ", avgD)

State = T.matrix("State")
Distance = T.matrix("Distances")

State = T.sum(State**2, axis=1)
Distance = T.sqrt((State + State[T.newaxis].T) - 2*T.dot(State, State.T))

get_mean_dist = theano.function([State], Distance)

dist_ = get_mean_dist(s)
print ("Theano dist: ", dist_)