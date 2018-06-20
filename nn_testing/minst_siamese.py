'''Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
os.environ['KERAS_BACKEND'] = "tensorflow"

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, Reshape
from keras.optimizers import RMSprop
from keras import backend as K

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    print("digit_indices: ", digit_indices)
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    print ("n: ", n)
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_pairs2(x):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pair1 = []
    pair2 = []
    labels = []
    n = x.shape[0] - 1
    indices = list(np.random.randint(low=0, high=n, size=n))
    for i in range(n):
        ### Identical pair
        i = indices[i]
        noise = np.random.normal(loc=0, scale=0.05, size=x[i].shape)
        x1 = [x[i] + noise]
        noise = np.random.normal(loc=0, scale=0.05, size=x[i].shape)
        x2 = [x[i] + noise]
        if (np.random.rand() > 0.5):
            pair1 += x1
            pair2 += x2
        else:
            pair1 += x2
            pair2 += x1
        ### Different pair
        z=i
        while (z == i): ## get result that is not the same
            z = np.random.randint(low=0, high=n)
        noise = np.random.normal(loc=0, scale=0.05, size=x[i].shape)
        x1 = [x[i] + noise]
        noise = np.random.normal(loc=0, scale=0.05, size=x[i].shape)
        x2 = [x[z] + noise]
        if (np.random.rand() > 0.5):
            pair1 += x1
            pair2 += x2
        else:
            pair1 += x2
            pair2 += x1
        
        labels += [1, 0]
    return np.array(pair1), np.array(pair2), np.array(labels)

def create_base_network(input, input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    net = Dense(128, activation='relu')(input)
    net = Dropout(0.2)(net)
    net = Dense(128, activation='relu')(net)
    net = Dropout(0.2)(net)
    net = Dense(128, activation='relu')(net)
    return net

def create_base_conv_network(input, input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    # net = Input((input_dim,))
    net = Reshape((28, 28, 1))(input)
    net = Conv2D(8, kernel_size=(4,4), strides=(1,1), activation='relu')(net)
    net = Dropout(0.1)(net)
    net = Conv2D(8, kernel_size=(4,4), strides=(1,1), activation='relu')(net)
    net = Dropout(0.1)(net)
    net = Dense(128, activation='relu')(net)
    return net


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
input_dim = 784
nb_epoch = 20

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
# tr_pairs, tr_y = create_pairs(X_train, digit_indices)
tr_pair1, tr_pair2, tr_y = create_pairs2(X_train)

digit_indices = [np.where(y_test == i)[0] for i in range(10)]
# te_pairs, te_y = create_pairs(X_test, digit_indices)
te_pair1, te_pair2, te_y = create_pairs2(X_test)

input_a = Input(shape=(input_dim,), name="Input_a")
input_b = Input(shape=(input_dim,), name="Input_b")
# network definition
base_network = create_base_network(input_a, input_dim)
# base_network = create_base_conv_network(input_a, input_dim)
base_network = Model(inputs=[input_a], outputs=base_network)


# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(inputs=[input_a, input_b], outputs=distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([tr_pair1, tr_pair2], tr_y,
          validation_data=([te_pair1, te_pair2], te_y),
          batch_size=128,
          epochs=nb_epoch)

# compute final accuracy on training and test sets
pred = model.predict([tr_pair1, tr_pair2])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pair1, te_pair2])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

print ("Predictions: ", pred)
print ("test y: ", te_y)

print ("Predictions: ", np.concatenate((np.reshape(te_y, (-1, 1)), pred), axis=1))