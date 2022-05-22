import numpy as np
# import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from model.LearningUtil import loglikelihood, loglikelihoodMEAN, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat, likelihood, loglikelihoodMEAN
from model.LearningUtil import loglikelihood, likelihood, likelihoodMEAN, kl, kl_D, entropy, flatgrad, zipsame, get_params_flat, setFromFlat
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras
from keras.models import Sequential, Model

# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.KERASAlgorithm import KERASAlgorithm

def eucl_dist_output_shape_seq(shapes):
    shape1, shape2 = shapes
    return shape1

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

def euclidean_distance(vects):
    x, y = vects
    return K.sum(K.square(x - y), axis=1, keepdims=True)

def euclidean_distance_(vects):
    x, y = vects
    return K.square(x - y)

def euclidean_distance_np(vects):
    x, y = vects
    return np.sum(np.square(x - y), axis=1, keepdims=True)

def l1_distance(vects):
    x, y = vects
    return K.sum(K.abs(x - y), axis=1, keepdims=True)

def l1_distance_np(vects):
    x, y = vects
    return np.sum(np.abs(x - y), axis=1, keepdims=True)

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    ####           Make these smaller               While making these bigger
    return K.mean((y_true * y_pred) + ((1 - y_true) * K.maximum(margin - y_pred, 0)))

def contrastive_loss_np(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    ####           Make these smaller               While making these bigger
    return np.mean((y_true * y_pred) + ((1 - y_true) * np.maximum(margin - y_pred, 0)))

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    cutoff = 0.5
    pred = predictions.ravel()
    # print ("pred: ", pred)
    ind_ = pred < cutoff
    ind_neg = pred >= cutoff
    
    # print ("ind_: ", ind_)
    ### positive label values that were within a distance of 0.5 of 0
    values_ = labels[ind_]
    ### negative label values that were within a distance of 0.5 of 1
    values_neg = 1.0 - labels[ind_neg]
    
    pos_error = np.fabs(pred[labels.ravel() > 0.5]) ### Distance these are from 0.0
    neg_error = np.fabs(1.0 - pred[labels.ravel() <= 0.5]) ### Distance these are from 1.0
    # print ("positive pair: ", pos_error)
    # print ("negative pair: ", neg_error)
    # print ("positive pair mean: ", np.mean(pred[labels.ravel() > 0.5]))
    # print ("negative pair mean: ", np.mean(pred[labels.ravel() <= 0.5]))
    ### What if all the labels are above 0.5... then neg will be nan...
    if ( len(neg_error) == 0):
        neg_error = [0]
    if ( len(pos_error) == 0):
        pos_error = [0]
    
    # print ("values_: ", values_)
    if (values_ == []): ### No values were close...
        return 0.0
    else:
        return np.concatenate((np.array(pos_error), np.array(neg_error)), axis=0)
    
def add_noise(noise_scale_, data, shape__=None):
    if shape__ is not None:
        shape_ = shape__
    else:
        shape_ = data.shape
    if (noise_scale_ > 0.001):
        return data + np.random.normal(loc=0, scale=noise_scale_, size=shape_)
    else:
        # print ("Not adding image noise")
        return data + np.zeros(shape_)

def create_sequences2(traj0, traj1, settings):
    '''Positive and negative sequence creation.
    Alternates between positive and negative pairs.
    produces N sequences from two
    
    Assume tr0 != tr1
    '''
    ### This a little hint that maybe some options are not perfect 1s or 0s.
    compare_adjustment = 0.0
    if ("imperfect_compare_offset" in settings):
        compare_adjustment = settings["imperfect_compare_offset"]
        # print ("compare_adjustment: ", compare_adjustment)
    noise_scale = 0.02
    target_noise_scale = 0.02
    if ("image_noise_scale" in settings):
        noise_scale = settings["image_noise_scale"]
        target_noise_scale = settings["image_noise_scale"]
    if ("target_noise_scale" in settings):
        target_noise_scale = settings["target_noise_scale"]
    sequences0 = []
    sequences1 = []
    targets_ = []
    for tr0, tr1 in zip(traj0, traj1): ### for each trajectory pair
        tar_shape = (len(tr0)-1, 1)
        if (len(tr0) == 1):
            tar_shape = (len(tr0), 1)
            ### Same trajectory but with noise
            sequences0.append(add_noise(noise_scale, tr0))
            sequences1.append(add_noise(noise_scale, tr0))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr1))
            sequences1.append(add_noise(noise_scale, tr1))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            ### Different trajectories
            sequences0.append(add_noise(noise_scale, tr0))
            sequences1.append(add_noise(noise_scale, tr1))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr1))
            sequences1.append(add_noise(noise_scale, tr0))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
        elif (len(tr0) == 2):
            
            tar_shape = (len(tr0), 1)
            ### Same clips
            sequences0.append(add_noise(noise_scale, tr0))
            sequences1.append(add_noise(noise_scale, tr0))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr0))
            sequences1.append(add_noise(noise_scale, tr0))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            sequences0.append(add_noise(noise_scale, tr1))
            sequences1.append(add_noise(noise_scale, tr1))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr1))
            sequences1.append(add_noise(noise_scale, tr1))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            """
            sequences0.append(add_noise(noise_scale, tr0))
            sequences1.append(add_noise(noise_scale, tr1))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            sequences0.append(add_noise(noise_scale, tr1))
            sequences1.append(add_noise(noise_scale, tr0))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            """
            
            ### clips with repeated frames
            sequences0.append(add_noise(noise_scale, [tr0[0]] + tr0[1:], shape__=tr1.shape ))
            sequences1.append(add_noise(noise_scale, [tr0[0]] + tr0[1:], shape__=tr1.shape))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            sequences0.append(add_noise(noise_scale, [tr1[0]] + tr1[1:], shape__=tr1.shape ))
            sequences1.append(add_noise(noise_scale, [tr1[0]] + tr1[1:], shape__=tr1.shape))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            ### reversed versions of the same trajectories
            sequences0.append(list(reversed(add_noise(noise_scale, tr1))))
            sequences1.append(add_noise(noise_scale, tr1))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(list(reversed(add_noise(noise_scale, tr1))))
            sequences1.append(add_noise(noise_scale, tr1))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            sequences0.append(list(reversed(add_noise(noise_scale, tr0))))
            sequences1.append(add_noise(noise_scale, tr0))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(list(reversed(add_noise(noise_scale, tr0))))
            sequences1.append(add_noise(noise_scale, tr0))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            # if ("include_agent_imitator_pairs" in settings
            #     and (settings["include_agent_imitator_pairs"] == True)):
                ### Versions of two different trajectories
            if (np.random.rand() > 0.5):
                sequences0.append(add_noise(noise_scale, tr0))
                sequences1.append(add_noise(noise_scale, tr1))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(add_noise(target_noise_scale, targets))
                
                sequences0.append(add_noise(noise_scale, tr1))
                sequences1.append(add_noise(noise_scale, tr0))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(add_noise(target_noise_scale, targets))
            else:
                sequences0.append(list(reversed(add_noise(noise_scale, tr0))))
                sequences1.append(list(reversed(add_noise(noise_scale, tr1))))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(add_noise(target_noise_scale, targets))
                
                sequences0.append(list(reversed(add_noise(noise_scale, tr1))))
                sequences1.append(list(reversed(add_noise(noise_scale, tr0))))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(add_noise(target_noise_scale, targets))                
            # print ("sequences0: ", np.array(sequences0).shape)
            
        else:
            ### basic for now
            
            ### Noisy versions of the same trajectories
            
            sequences0.append(add_noise(noise_scale, tr0[1:]))
            sequences1.append(add_noise(noise_scale, tr0[1:]))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
#             sequences0.append(add_noise(noise_scale, tr1[1:]))
#             sequences1.append(add_noise(noise_scale, tr1[1:]))
#             targets = np.ones(tar_shape)
#             targets_.append(add_noise(target_noise_scale, targets))
            
            sequences0.append(add_noise(noise_scale, tr0[:-1]))
            sequences1.append(add_noise(noise_scale, tr0[:-1]))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
#             sequences0.append(add_noise(noise_scale, tr1[:-1]))
#             sequences1.append(add_noise(noise_scale, tr1[:-1]))
#             targets = np.ones(tar_shape)
#             targets_.append(add_noise(target_noise_scale, targets))
            
            ### Out of sync versions of the same trajectories
            sequences0.append(add_noise(noise_scale, tr0[1:]))
            sequences1.append(add_noise(noise_scale, tr0[:-1]))
            targets = np.ones(tar_shape) - compare_adjustment
            targets[0] = 0
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr0[1:]))
            sequences1.append(add_noise(noise_scale, tr0[:-1] ))
            targets = np.ones(tar_shape) - compare_adjustment
            targets[-1] = 0
            targets_.append(add_noise(target_noise_scale, targets))
            
            ### Out of sync versions of the same trajectories
            sequences0.append(add_noise(noise_scale, tr1[1:]))
            sequences1.append(add_noise(noise_scale, tr1[:-1]))
            targets = np.ones(tar_shape) - compare_adjustment
            targets[0] = 0
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr1[1:]))
            sequences1.append(add_noise(noise_scale, tr1[:-1] ))
            targets = np.ones(tar_shape) - compare_adjustment
            targets[-1] = 0
            targets_.append(add_noise(target_noise_scale, targets))
            
            
            ### reversed versions of the same trajectories
            indicies = np.array(list(range(len(tr1))))
            rand_ind = np.array(list(reversed(range(len(tr1)))))
            # print ("indicies: ", indicies)
            # print ("rand_ind: ", rand_ind)
            # print ("rand_ind == indicies[1:]: ", rand_ind[1:] == indicies[1:])
#             if (np.random.rand() > 0.5):
#                 sequences0.append(list(reversed(add_noise(noise_scale, tr1[1:]))))
#                 sequences1.append(add_noise(noise_scale, tr1[:-1]))
#                 targets = np.zeros(tar_shape)
#                 targets_.append(add_noise(target_noise_scale, targets))
#             else:
#                 sequences0.append(add_noise(noise_scale, tr1[:-1]))
#                 sequences1.append(list(reversed(add_noise(noise_scale, tr1[:-1]))))
#                 targets = np.array([[int(g)] for g in (rand_ind[:-1] == indicies[:-1])])
#                 targets_.append(add_noise(target_noise_scale, targets))
#             
#             if (np.random.rand() > 0.5):
#                 sequences0.append(list(reversed(add_noise(noise_scale, tr0[1:]))))
#                 sequences1.append(add_noise(noise_scale, tr0[:-1]))
#                 targets = np.zeros(tar_shape)
#                 targets_.append(add_noise(target_noise_scale, targets))
#             else:
#                 sequences0.append(add_noise(noise_scale, tr0[:-1]))
#                 sequences1.append(list(reversed(add_noise(noise_scale, tr0[:-1]))))
#                 targets = np.array([[int(g)] for g in (rand_ind[:-1] == indicies[:-1])])
#                 targets_.append(add_noise(target_noise_scale, targets))
            
            ### Random frozen frame versions of sequences
#             if (np.random.rand() > 0.5):
#                 sequences0.append(add_noise(noise_scale, tr1[1:]))
#                 sequences1.append(add_noise(noise_scale, [tr1[np.random.choice(range(len(tr1)))] * len(tr1[1:])], shape__=tr0[1:].shape))
#                 targets = np.zeros(tar_shape) + compare_adjustment
#                 targets_.append(add_noise(target_noise_scale, targets))
#             else:
#                 sequences0.append(add_noise(noise_scale, tr1[:-1] ))
#                 sequences1.append(add_noise(noise_scale, [tr1[np.random.choice(range(len(tr1)))] * len(tr1[1:])], shape__=tr0[1:].shape ))
#                 targets = np.zeros(tar_shape) + compare_adjustment
#                 targets_.append(add_noise(target_noise_scale, targets))
#             #
#             if (np.random.rand() > 0.5):
#                 sequences0.append(add_noise(noise_scale, [tr1[np.random.choice(range(len(tr1)))] * len(tr1[1:])], shape__=tr0[1:].shape ))
#                 sequences1.append(add_noise(noise_scale, tr1[1:]))
#                 targets = np.zeros(tar_shape) + compare_adjustment
#                 targets_.append(add_noise(target_noise_scale, targets))
#             #
#             else:
#                 sequences0.append(add_noise(noise_scale, [tr1[np.random.choice(range(len(tr1)))] * len(tr1[1:])], shape__=tr0[1:].shape ))
#                 sequences1.append(add_noise(noise_scale, tr1[:-1] ))
#                 targets = np.zeros(tar_shape) + compare_adjustment
#                 targets_.append(add_noise(target_noise_scale, targets))
#             #
#             if (np.random.rand() > 0.5):
#                 sequences0.append(add_noise(noise_scale, tr0[1:]))
#                 sequences1.append(add_noise(noise_scale, [tr0[np.random.choice(range(len(tr0)))] * len(tr0[1:])], shape__=tr0[1:].shape ))
#                 targets = np.zeros(tar_shape) + compare_adjustment
#                 targets_.append(add_noise(target_noise_scale, targets))
#             #
#             else:
#                 sequences0.append(add_noise(noise_scale, tr0[:-1]))
#                 sequences1.append(add_noise(noise_scale, [tr0[np.random.choice(range(len(tr0)))] * len(tr1[1:])], shape__=tr0[1:].shape ))
#                 targets = np.zeros(tar_shape) + compare_adjustment
#                 targets_.append(add_noise(target_noise_scale, targets))
#             #
#             if (np.random.rand() > 0.5):
#                 sequences0.append(add_noise(noise_scale, [tr0[np.random.choice(range(len(tr0)))] * len(tr0[1:])], shape__=tr0[1:].shape ))
#                 sequences1.append(add_noise(noise_scale, tr0[1:]))
#                 targets = np.zeros(tar_shape) + compare_adjustment
#                 targets_.append(add_noise(target_noise_scale, targets))
#             #
#             else:
#                 sequences0.append(add_noise(noise_scale, [tr0[np.random.choice(range(len(tr0)))] * len(tr0[1:])], shape__=tr0[1:].shape ))
#                 sequences1.append(add_noise(noise_scale, tr0[:-1] ))
#                 targets = np.zeros(tar_shape) + compare_adjustment
#                 targets_.append(add_noise(target_noise_scale, targets))


            sequences0.append(add_noise(noise_scale, tr1[1:]))
            sequences1.append(add_noise(noise_scale, tr0[1:]))
            targets = np.zeros(tar_shape) + compare_adjustment
            targets_.append(add_noise(target_noise_scale, targets))
            
            sequences0.append(add_noise(noise_scale, tr0[1:]))
            sequences1.append(add_noise(noise_scale, tr1[1:]))
            targets = np.zeros(tar_shape) + compare_adjustment
            targets_.append(add_noise(target_noise_scale, targets))
            
            sequences0.append(add_noise(noise_scale, tr1[:-1]))
            sequences1.append(add_noise(noise_scale, tr0[1:]))
            targets = np.zeros(tar_shape) + compare_adjustment
            targets_.append(add_noise(target_noise_scale, targets))
             
            sequences0.append(add_noise(noise_scale, tr0[:-1]))
            sequences1.append(add_noise(noise_scale, tr1[1:]))
            targets = np.zeros(tar_shape) + compare_adjustment
            targets_.append(add_noise(target_noise_scale, targets))
            
            sequences0.append(add_noise(noise_scale, tr1[:-1]))
            sequences1.append(add_noise(noise_scale, tr0[:-1]))
            targets = np.zeros(tar_shape) + compare_adjustment
            targets_.append(add_noise(target_noise_scale, targets))
            
            sequences0.append(add_noise(noise_scale, tr0[:-1]))
            sequences1.append(add_noise(noise_scale, tr1[:-1]))
            targets = np.zeros(tar_shape) + compare_adjustment
            targets_.append(add_noise(target_noise_scale, targets))

            ### Randomly shuffled sequences
            indicies = range(len(tr1))
            # print ("indicies: ", indicies)
            # print ("choice: ", np.random.choice(indicies, len(tr1[1:])))
#             if (np.random.rand() > 0.5):
#                 rand_ind = np.random.choice(indicies, len(tr1[1:]))
#                 sequences0.append(add_noise(noise_scale, tr1[1:]))
#                 sequences1.append(add_noise(noise_scale, np.array(tr1)[rand_ind]))
#                 targets = np.array([[int(g)] for g in (rand_ind == indicies[1:])])
#                 targets_.append(add_noise(target_noise_scale, targets))
#             else:
#                 rand_ind = np.random.choice(indicies, len(tr0[1:]))
#                 sequences0.append(add_noise(noise_scale, tr0[1:]))
#                 sequences1.append(add_noise(noise_scale, np.array(tr0)[rand_ind] ))
#                 targets = np.array([[int(g)] for g in (rand_ind == indicies[1:])])
#                 targets_.append(add_noise(target_noise_scale, targets))
#                 
#             if (np.random.rand() > 0.5):
#                 rand_ind0 = np.random.choice(indicies, len(tr1[1:]))
#                 rand_ind1 = np.random.choice(indicies, len(tr1[1:]))
#                 sequences0.append(add_noise(noise_scale, np.array(tr1)[rand_ind0] ))
#                 sequences1.append(add_noise(noise_scale, np.array(tr1)[rand_ind1] ))
#                 targets = np.zeros(tar_shape)
#                 targets = np.array([[int(g)] for g in (rand_ind0 == rand_ind1)])
#                 targets_.append(add_noise(target_noise_scale, targets))
#             else:
#                 rand_ind0 = np.random.choice(indicies, len(tr0[1:]))
#                 rand_ind1 = np.random.choice(indicies, len(tr0[1:]))
#                 sequences0.append(add_noise(noise_scale, np.array(tr0)[rand_ind0] ))
#                 sequences1.append(add_noise(noise_scale, np.array(tr0)[rand_ind1] ))
#                 targets = np.array([[int(g)] for g in (rand_ind0 == rand_ind1)])
#                 targets_.append(add_noise(target_noise_scale, targets))
                
    return sequences0, sequences1, targets_


def create_advisarial_sequences(traj0, traj1, settings):
    import random
    '''Positive and negative sequence creation.
    Alternates between positive and negative pairs.
    produces N sequences from two
    
    Assume tr0 != tr1
    '''
    ### This a little hint that maybe some options are not perfect 1s or 0s.
    compare_adjustment = 0.0
    if ("imperfect_compare_offset" in settings):
        compare_adjustment = settings["imperfect_compare_offset"]
        # print ("compare_adjustment: ", compare_adjustment)
    noise_scale = 0.02
    target_noise_scale = 0.02
    if ("image_noise_scale" in settings):
        noise_scale = settings["image_noise_scale"]
        target_noise_scale = settings["image_noise_scale"]
    if ("target_noise_scale" in settings):
        target_noise_scale = settings["target_noise_scale"]
    sequences0 = []
    sequences1 = []
    targets_ = []
    indx = list(range(len(traj0)))
    
    for i in range(len(traj0)): ### for each trajectory pair
        tr0 = traj0[i]
        tr1 = traj1[i]
        tar_shape = (len(tr0)-1, 1)
        if (len(tr0) == 1):
            tar_shape = (len(tr0), 1)
            ### same trajectories
            sequences0.append(add_noise(noise_scale, tr0))
            sequences1.append(add_noise(noise_scale, tr0))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr1))
            sequences1.append(add_noise(noise_scale, tr1))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            ### Advisarial trajectories
            sequences0.append(add_noise(noise_scale, tr0))
            sequences1.append(add_noise(noise_scale, traj0[random.sample(set(indx), 1)[0]]))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, traj1[random.sample(set(indx), 1)[0]]))
            sequences1.append(add_noise(noise_scale, tr0))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
        elif (len(tr0) == 2):
            
            tar_shape = (len(tr0), 1)
            ### Same clips
            sequences0.append(add_noise(noise_scale, tr0))
            sequences1.append(add_noise(noise_scale, tr0))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr1))
            sequences1.append(add_noise(noise_scale, tr1))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            """
            sequences0.append(add_noise(noise_scale, tr0))
            sequences1.append(add_noise(noise_scale, tr1))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            sequences0.append(add_noise(noise_scale, tr1))
            sequences1.append(add_noise(noise_scale, tr0))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            """
            
            ### Advisarial trajectories
            sequences0.append(add_noise(noise_scale, tr0))
            sequences1.append(add_noise(noise_scale, traj1[random.sample(set(indx), 1)[0]]))
            targets = np.zeros(tar_shape) + compare_adjustment
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr1))
            sequences1.append(add_noise(noise_scale, traj0[random.sample(set(indx), 1)[0]]))
            targets = np.zeros(tar_shape) + compare_adjustment
            targets_.append(add_noise(target_noise_scale, targets))
            
        else:
            
            # if (pos_batch):
            ### Noisy versions of the same trajectories
#             sequences0.append(add_noise(noise_scale, tr0[1:]))
#             sequences1.append(add_noise(noise_scale, tr0[:-1]))
#             targets = np.ones(tar_shape)
#             targets_.append(add_noise(target_noise_scale, targets))
#             sequences0.append(add_noise(noise_scale, tr1[1:]))
#             sequences1.append(add_noise(noise_scale, tr1[:-1]))
#             targets = np.ones(tar_shape)
#             targets_.append(add_noise(target_noise_scale, targets))
            
            """
            sequences0.append(add_noise(noise_scale, tr0[:-1]))
            sequences1.append(add_noise(noise_scale, tr0[:-1]))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr1[:-1]))
            sequences1.append(add_noise(noise_scale, tr1[:-1]))
            targets = np.ones(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            """ 
            
            ### trajectories from sim and real env should be similar
            
            if (np.random.rand() > 0.5): 
                sequences0.append(add_noise(noise_scale, tr0[1:]))
                sequences1.append(add_noise(noise_scale, traj0[random.sample(set(indx) - set([i]), 1)[0]][1:]))
                targets = np.ones(tar_shape) - compare_adjustment
                targets_.append(add_noise(target_noise_scale, targets))
                sequences0.append(add_noise(noise_scale, tr1[1:]))
                sequences1.append(add_noise(noise_scale, traj1[random.sample(set(indx) - set([i]), 1)[0]][1:]))
                targets = np.ones(tar_shape) - compare_adjustment
                targets_.append(add_noise(target_noise_scale, targets))
            else:
                sequences0.append(add_noise(noise_scale, tr0[:-1]))
                sequences1.append(add_noise(noise_scale, traj0[random.sample(set(indx) - set([i]), 1)[0]][:-1]))
                targets = np.ones(tar_shape) - compare_adjustment
                targets_.append(add_noise(target_noise_scale, targets))
                sequences0.append(add_noise(noise_scale, tr1[:-1]))
                sequences1.append(add_noise(noise_scale, traj1[random.sample(set(indx) - set([i]), 1)[0]][:-1]))
                targets = np.ones(tar_shape) - compare_adjustment
                targets_.append(add_noise(target_noise_scale, targets))

            ### trajectories from sim and real env should be similar
            sequences0.append(add_noise(noise_scale, tr0[1:]))
            sequences1.append(add_noise(noise_scale, traj0[random.sample(set(indx), 1)[0]][:-1]))
            targets = np.ones(tar_shape) - compare_adjustment
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr1[1:]))
            sequences1.append(add_noise(noise_scale, traj1[random.sample(set(indx), 1)[0]][:-1]))
            targets = np.ones(tar_shape) - compare_adjustment
            targets_.append(add_noise(target_noise_scale, targets))
            
            
            # else:
#             ### Versions of two different adversarial trajectories
#             advisarial_swap_prob = 0.999
#             sequences0.append(add_noise(noise_scale, tr0[1:]))
#             sequences1.append(add_noise(noise_scale, tr1[1:]))
#             targets = np.zeros(tar_shape)
#             targets_.append(add_noise(target_noise_scale, targets))
#             sequences0.append(add_noise(noise_scale, tr0[:-1]))
#             sequences1.append(add_noise(noise_scale, tr1[:-1]))
#             targets = np.zeros(tar_shape)
#             targets_.append(add_noise(target_noise_scale, targets))
            ### Versions of two different adversarial trajectories from different classes
            sequences0.append(add_noise(noise_scale, tr0[1:]))
            sequences1.append(add_noise(noise_scale, traj1[random.sample(set(indx), 1)[0]][1:]))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr1[1:]))
            sequences1.append(add_noise(noise_scale, traj0[random.sample(set(indx), 1)[0]][1:]))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            sequences0.append(add_noise(noise_scale, tr0[1:]))
            sequences1.append(add_noise(noise_scale, traj1[random.sample(set(indx), 1)[0]][:-1]))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, tr1[1:]))
            sequences1.append(add_noise(noise_scale, traj0[random.sample(set(indx), 1)[0]][:-1]))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            
            ### More Out of sync versions of two adversarial trajectories
            """
            sequences0.append(add_noise(noise_scale, traj0[random.sample(set(indx), 1)[0]][1:]))
            sequences1.append(add_noise(noise_scale, tr1[:-1]))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            sequences0.append(add_noise(noise_scale, traj0[random.sample(set(indx), 1)[0]][:-1]))
            sequences1.append(add_noise(noise_scale, tr1[1:]))
            targets = np.zeros(tar_shape)
            targets_.append(add_noise(target_noise_scale, targets))
            """
            
    # print ("Created advisarial trajectories: ")
    
    return sequences0, sequences1, targets_

def create_multitask_sequences(traj0, traj1, task_ids, settings):
    '''Positive and negative sequence creation.
    Alternates between positive and negative pairs.
    produces N sequences from two
    
    ### We want the average for targets_ to be near .5 so the loss is balanced
    
    class ids are stored in task_ids
    '''
    noise_scale = 0.02
    target_noise_scale = 0.02
    if ("image_noise_scale" in settings):
        noise_scale = settings["image_noise_scale"]
        target_noise_scale = settings["image_noise_scale"]
    compare_adjustment = 0.0
    if ("imperfect_compare_offset" in settings):
        compare_adjustment = settings["imperfect_compare_offset"]
    use_agent_multitask_data = False
    if ("use_agent_multitask_data" in settings):
        compare_adjustment = settings["use_agent_multitask_data"]
    sequences0 = []
    sequences1 = []
    targets_ = []
    poss = 0 ### Keep track of the number of positive vs negative pairs
    ### Generating Multi-task target labels
    for i in range(len(traj0)):
    # for tr0, task_tr0 in zip(traj0, task_ids): ### for each trajectory pair
        tar_shape = (len(traj0[i]), 1)
        
        for j in range(len(traj0)):
        # for tr1, task_tr1 in zip(traj0, task_ids): ### for each trajectory pair
        
            ### Noisy versions of imitation trajectories
            sequences0.append(traj0[i] + np.random.normal(loc=0, scale=noise_scale, size=traj0[i].shape))
            sequences1.append(traj0[j] + np.random.normal(loc=0, scale=noise_scale, size=traj0[j].shape))
            
            ### Noisy versions of agent trajectories
            if (use_agent_multitask_data):
                sequences0.append(traj1[i] + np.random.normal(loc=0, scale=noise_scale, size=traj0[i].shape))
                sequences1.append(traj1[j] + np.random.normal(loc=0, scale=noise_scale, size=traj0[j].shape))
            # print ("task_tr0[0][0] == task_tr1[0][0]", task_tr0[0][0], " == ", task_tr1[0][0])
            # print ("settings['worker_to_task_mapping'][task_tr0[0]] == settings['worker_to_task_mapping'][task_tr1[0]]", 
            #        settings["worker_to_task_mapping"][task_tr0[0][0]]," == ", settings["worker_to_task_mapping"][task_tr1[0][0]])
            ### This logic is to make sure these batches are balanced wrt positives and negatives.
            if ("ask_env_for_multitask_id" in settings 
                and (settings["ask_env_for_multitask_id"])):
                if  (task_ids[i][0][0] == task_ids[j][0][0]):
                    if ( i == j ): ### same trajectory
                        targets = np.ones(tar_shape)
                        poss = poss + 1
                    else:
                        targets = np.ones(tar_shape) - compare_adjustment
                        poss = poss + 1
                else:
                    if (poss >= 0):
                        targets = np.zeros(tar_shape)
                        poss = poss + -1
                    else:
                        continue
            elif (settings["worker_to_task_mapping"][task_ids[i][0][0]] == settings["worker_to_task_mapping"][task_ids[j][0][0]]): ### same task
                if ( i == j ): ### same trajectory
                    targets = np.ones(tar_shape)
                    poss = poss + 1
                else:
                    targets = np.ones(tar_shape) - compare_adjustment
                    poss = poss + 1
            else:
                if (poss >= 0):
                    targets = np.zeros(tar_shape)
                    poss = poss + -1
                else:
                    continue
            # print ("task_ids[i][0][0]: ", task_ids[i][0][0], " task_ids[j][0][0]: ", task_ids[j][0][0])
            targets_.append(add_noise(target_noise_scale, targets))
            if (use_agent_multitask_data):
                targets_.append(add_noise(target_noise_scale, targets))
                
    ### We want the average for targets_ to be near .5
    # print ("multitask targets mean", np.mean(targets_), " count: ", np.array(targets_).shape, " traj0 shape: ", np.array(traj0).shape, " poss: ", poss)
        
    return sequences0, sequences1, targets_

        
def create_pairs2(x, settings):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    target_noise_scale = 0.05
    compare_adjustment = 0.0
    if ("imperfect_compare_offset" in settings):
        compare_adjustment = settings["imperfect_compare_offset"]
    noise_scale = 0.02
    if ("image_noise_scale" in settings):
        noise_scale = settings["image_noise_scale"]
        target_noise_scale = settings["image_noise_scale"]
    pair1 = []
    pair2 = []
    labels = []
    n = x.shape[0] - 1
    indices = list(np.random.randint(low=0, high=n, size=n))
    for i in range(n):
        ### Identical pair
        i = indices[i]
        # noise = np.random.normal(loc=0, scale=noise_scale, size=x[i].shape)
        x1 = [add_noise(noise_scale, x[i])]
        # noise = np.random.normal(loc=0, scale=noise_scale, size=x[i].shape)
        x2 = [add_noise(noise_scale, x[i])]
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
        # noise = np.random.normal(loc=0, scale=noise_scale, size=x[i].shape)
        x1 = [add_noise(noise_scale, x[i] )]
        # noise = np.random.normal(loc=0, scale=noise_scale, size=x[i].shape)
        x2 = [add_noise(noise_scale, x[z] )]
        if (np.random.rand() > 0.5):
            pair1 += x1
            pair2 += x2
        else:
            pair1 += x2
            pair2 += x1
        labels += [np.clip(1 + np.random.normal(loc=0, scale=target_noise_scale, size=1), 0.01, 0.98),
                    np.clip(0 + np.random.normal(loc=0, scale=target_noise_scale, size=1), 0.01, 0.98)]
    return np.array(pair1), np.array(pair2), np.array(labels)

class SiameseNetwork(KERASAlgorithm):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        super(SiameseNetwork,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        self._model = model
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-6
        
        self._distance_func = euclidean_distance
        self._distance_func_np = euclidean_distance_np
        if ( "fd_distance_function" in self.getSettings()
             and (self.getSettings()["fd_distance_function"] == "l1")):
            print ("Using ", self.getSettings()["fd_distance_function"], " distance metric for siamese network.")
            self._distance_func = l1_distance
            self._distance_func_np = l1_distance_np
        condition_reward_on_result_state = False
        self._train_combined_loss = False

        inputs_ = [self._model.getStateSymbolicVariable()] 
        self._model._forward_dynamics_net = Model(inputs=inputs_, outputs=self._model._forward_dynamics_net)
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("FD Net summary: ", self._model._forward_dynamics_net.summary())
        
        if ("force_use_actor_state_for_critic" in self._settings
            and (self._settings["force_use_actor_state_for_critic"] == True)):
            inputs_ = [self._model.getStateSymbolicVariable()]
        else:  
            inputs_ = [self._model.getResultStateSymbolicVariable()] 
        self._model._reward_net = Model(inputs=inputs_, outputs=self._model._reward_net)
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("FD Reward Net summary: ", self._model._reward_net.summary())

        self._modelTarget = None
        SiameseNetwork.compile(self)
    
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        
        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        state_copy = keras.layers.Input(shape=keras.backend.int_shape(self._model.getStateSymbolicVariable())[1:], name="State_2")
        result_state_copy = keras.layers.Input(shape=keras.backend.int_shape(self._model.getResultStateSymbolicVariable())[1:]
                                                                              , name="ResultState_2"
                                                                              )
        print ("*** self._model.getStateSymbolicVariable() shape: ", repr(keras.backend.int_shape(self._model.getStateSymbolicVariable())))
        print ("*** self._model.getStateSymbolicVariable() shape: ", repr(self._model.getStateSymbolicVariable()))
        print ("*** self._model.getResultStateSymbolicVariable() shape: ", repr(keras.backend.int_shape(self._model.getResultStateSymbolicVariable())))
        print ("*** self._model.getResultStateSymbolicVariable() shape: ", repr(self._model.getResultStateSymbolicVariable()))
        processed_a = self._model._forward_dynamics_net(self._model.getStateSymbolicVariable())
        self._model.processed_a = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=processed_a)
        processed_b = self._model._forward_dynamics_net(state_copy)
        self._model.processed_b = Model(inputs=[state_copy], outputs=processed_b)
        if ( "return_rnn_sequence" in self.getSettings()
             and (self.getSettings()["return_rnn_sequence"])):
                
            processed_a_r_seq , processed_a_r, processed_a_r_c  = self._model._reward_net(self._model.getResultStateSymbolicVariable())
            processed_b_r_seq , processed_b_r, processed_b_r_c = self._model._reward_net(result_state_copy)
            
            print ("processed_a_r_c: ", repr(processed_a_r_c))
            
            if ("condition_on_rnn_internal_state" in self.getSettings()
                and (self.getSettings()["condition_on_rnn_internal_state"] == True)):
                processed_a_r = keras.layers.concatenate(inputs=[processed_a_r, processed_a_r_c], axis=1)
                processed_b_r = keras.layers.concatenate(inputs=[processed_b_r, processed_b_r_c], axis=1)

            encode_input__ = keras.layers.Input(shape=keras.backend.int_shape(processed_b_r)[1:]
                                                                              , name="encoding_2"
                                                                              )
            last_dense = keras.layers.Dense(64, activation = 'sigmoid')(encode_input__)
            self._last_dense = Model(inputs=[encode_input__], outputs=last_dense)
            
            processed_b_r = self._last_dense(processed_b_r)
            processed_a_r = self._last_dense(processed_a_r)
            

            self._model.processed_a_r = Model(inputs=[self._model.getResultStateSymbolicVariable()], outputs=processed_a_r)
            self._model.processed_b_r = Model(inputs=[result_state_copy], outputs=processed_b_r)

            processed_b_r_seq = keras.layers.TimeDistributed(self._last_dense)(processed_b_r_seq)
            processed_a_r_seq = keras.layers.TimeDistributed(self._last_dense)(processed_a_r_seq)
            # distance_r_weighted_seq = keras.layers.TimeDistributed(self._distance_weighting_)(distance_r_seq)
            
            self._model.processed_a_r_seq = Model(inputs=[self._model.getResultStateSymbolicVariable()], outputs=processed_a_r_seq)
            self._model.processed_b_r_seq = Model(inputs=[result_state_copy], outputs=processed_b_r_seq)
        else:
            if ("condition_on_rnn_internal_state" in self.getSettings()
                and (self.getSettings()["condition_on_rnn_internal_state"] == True)):
                _, processed_a_r, processed_a_r_c  = self._model._reward_net(self._model.getResultStateSymbolicVariable())
                _, processed_b_r, processed_b_r_c = self._model._reward_net(result_state_copy)
                processed_a_r = keras.layers.concatenate(inputs=[processed_a_r, processed_a_r_c], axis=1)
                processed_b_r = keras.layers.concatenate(inputs=[processed_b_r, processed_b_r_c], axis=1)
                
                encode_input__ = keras.layers.Input(shape=keras.backend.int_shape(processed_b_r)[1:]
                                                                              , name="encoding_2"
                                                                              )
                last_dense = keras.layers.Dense(64, activation = 'sigmoid')(encode_input__)
                self._last_dense = Model(inputs=[encode_input__], outputs=last_dense)
                
                processed_b_r = self._last_dense(processed_b_r)
                processed_a_r = self._last_dense(processed_a_r)
                
            else:
                processed_a_r = self._model._reward_net(self._model.getResultStateSymbolicVariable())
                processed_b_r = self._model._reward_net(result_state_copy)
            
            self._model.processed_a_r = Model(inputs=[self._model.getResultStateSymbolicVariable()], outputs=processed_a_r)
            self._model.processed_b_r = Model(inputs=[result_state_copy], outputs=processed_b_r)
        
        distance_fd = keras.layers.Lambda(self._distance_func, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        distance_r = keras.layers.Lambda(self._distance_func, output_shape=eucl_dist_output_shape)([processed_a_r, processed_b_r])

        self._model._forward_dynamics_net = Model(inputs=[self._model.getStateSymbolicVariable()
                                                          ,state_copy 
                                                          ]
                                                  , outputs=distance_fd
                                                  )
        
        if (("train_lstm_fd_and_reward_together" in self._settings)
            and (self._settings["train_lstm_fd_and_reward_together"] == True)):
            self._model._reward_net = Model(inputs=[self._model.getResultStateSymbolicVariable()
                                                          ,result_state_copy
                                                          ]
                                                          , outputs=distance_r
                                                          )
        else:
            self._model._reward_net = Model(inputs=[self._model.getResultStateSymbolicVariable()
                                                          ,result_state_copy
                                                          ]
                                                          , outputs=distance_r
                                                          )

        # print ("encode_input__: ", repr(encode_input__))
        # distance_r_weighted = keras.layers.Dense(64, activation = 'sigmoid')(encode_input__)
        # self._distance_weighting_ = Model(inputs=[encode_input__], outputs=distance_r_weighted)
        # distance_r_weighted = self._distance_weighting_(distance_r)
        # print ("distance_r_weighted: ", repr(distance_r_weighted))
        
        if ( "return_rnn_sequence" in self.getSettings()
             and (self.getSettings()["return_rnn_sequence"])):
            distance_r_seq = keras.layers.Lambda(self._distance_func, output_shape=eucl_dist_output_shape_seq)([processed_a_r_seq, processed_b_r_seq])
            print ("distance_r_seq: ", repr(distance_r_seq))
            # distance_r_weighted_seq = keras.layers.TimeDistributed(self._distance_weighting_)(distance_r_seq)
            # print ("distance_r_weighted_seq: ", repr(distance_r_weighted_seq))
            self._model._reward_net_seq = Model(inputs=[self._model.getResultStateSymbolicVariable()
                                                              ,result_state_copy
                                                              ]
                                                              , outputs=distance_r_seq
                                                              )

        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=2.5)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
        self._model._forward_dynamics_net.compile(loss=contrastive_loss, optimizer=sgd)

        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=2.5)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
        self._model._reward_net.compile(loss=contrastive_loss, optimizer=sgd)
        
        self._contrastive_loss = K.function([self._model.getStateSymbolicVariable(), 
                                             state_copy,
                                             K.learning_phase()], 
                                            [distance_fd])
        
        self._contrastive_loss_r = K.function([self._model.getResultStateSymbolicVariable(), 
                                             result_state_copy,
                                             K.learning_phase()], 
                                            [distance_r])
        # self.reward = K.function([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable(), K.learning_phase()], [self._reward])
        
    def reset(self):
        """
            Reset any state for the agent model
        """
        self._model.reset()
        self._model._reward_net.reset_states()
        self._model._forward_dynamics_net.reset_states()
        self._model.processed_a.reset_states()
        self._model.processed_b.reset_states()
        self._model.processed_a_r.reset_states()
        self._model.processed_b_r.reset_states()
        if not (self._modelTarget is None):
            self._modelTarget.reset()
            
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model._forward_dynamics_net.get_weights()))
        params.append(copy.deepcopy(self._model._reward_net.get_weights()))
        
        if ( "return_rnn_sequence" in self.getSettings()
             and (self.getSettings()["return_rnn_sequence"])):
            params.append(copy.deepcopy(self._model._reward_net_seq.get_weights()))
                
        return params
    
    def setNetworkParameters(self, params):
        self._model._forward_dynamics_net.set_weights(params[0])
        self._model._reward_net.set_weights(params[1])
        if ( "return_rnn_sequence" in self.getSettings()
             and (self.getSettings()["return_rnn_sequence"])):
            self._model._reward_net_seq.set_weights(params[2])
        
    def setGradTarget(self, grad):
        self._fd_grad_target_shared.set_value(grad)
        
    def getGrads(self, states, actions, result_states, v_grad=None, alreadyNormed=False):
        if ( alreadyNormed == False ):
            states = np.array(norm_state(states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            result_states = np.array(norm_state(result_states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        # result_states = np.array(result_states, dtype=self.getSettings()['float_type'])
        # self.setData(states, actions, result_states)
        # if (v_grad != None):
        # print ("states shape: ", states.shape, " actions shape: ", actions.shape, " v_grad.shape: ", v_grad.shape)
        self.setGradTarget(v_grad)
        # print ("states shape: ", states.shape, " actions shape: ", actions.shape)
        # grad = self._get_grad([states, actions])[0]
        grad = np.zeros_like(states)
        # print ("grad: ", grad)
        return grad
    
    def getRewardGrads(self, states, actions, alreadyNormed=False):
        # states = np.array(states, dtype=self.getSettings()['float_type'])
        # actions = np.array(actions, dtype=self.getSettings()['float_type'])
        if ( alreadyNormed is False ):
            states = np.array(norm_state(states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            # rewards = np.array(norm_state(rewards, self._reward_bounds), dtype=self.getSettings()['float_type'])
        # self.setData(states, actions)
        return self._get_grad_reward([states, actions, 0])[0]
    
    def updateTargetModel(self):
        pass
                
    def train(self, states, actions, result_states, rewards, falls=None, 
              updates=1, batch_size=None, p=1, lstm=True, datas=None, trainInfo=None):
        """
            states will come for the agent and
            results_states can come from the imitation agent
        """
        # print ("fd: ", self)
        # print ("state length: ", len(self.getStateBounds()[0]))
        self.reset()
        states_ = states
        if ('anneal_learning_rate' in self.getSettings()
            and (self.getSettings()['anneal_learning_rate'] == True)):
            K.set_value(self._model._forward_dynamics_net.optimizer.lr, np.float32(self.getSettings()['fd_learning_rate']) * p)
        if ("replace_next_state_with_imitation_viz_state" in self.getSettings()
            and (self.getSettings()["replace_next_state_with_imitation_viz_state"] == True)):
            states_ = np.concatenate((states_, result_states), axis=0)
        if (((("train_LSTM_FD" in self._settings)
                and (self._settings["train_LSTM_FD"] == True))
            or
            (("train_LSTM_Reward" in self._settings)
                and (self._settings["train_LSTM_Reward"] == True))
            ) 
            and lstm):
            ### result states can be from the imitation agent.
            # print ("falls: ", falls)
            if (falls is None):
                sequences0, sequences1, targets_ = create_sequences(states, result_states, self._settings)
            else:
                sequences0, sequences1, targets_ = create_multitask_sequences(states, result_states, datas["task_id"], self._settings)
            sequences0 = np.array(sequences0)
            # print ("sequences0 shape: ", sequences0.shape)
            sequences1 = np.array(sequences1)
            targets_ = np.array(targets_)
            
            if ( "add_label_noise" in self._settings):
                if (np.random.rand() < self._settings["add_label_noise"]):
                    # print ("targets_[0]: ", targets_[0])
                    targets_ = 1.0 - targets_ ### Invert labels
                    # print ("Inverting label values this time")
                    # print ("targets_[0]: ", targets_[0])
            # print ("targets_ shape: ", targets_.shape)
            # te_pair1, te_pair2, te_y = seq
            # score = self._model._forward_dynamics_net.train_on_batch([sequences0, sequences1], targets_)
            loss_ = []
            if ("train_LSTM_FD_stateful" in self._settings
                and (self._settings["train_LSTM_FD_stateful"] == True)
                # and False
                ):
                for k in range(sequences0.shape[1]):
                    ### shaping data
                    x0 = np.array(sequences0[:,[k]])
                    x1 = np.array(sequences1[:,[k]])
                    y0 = np.array(targets_[:,k]) ### For now reduce the dimensionality of the target because my nets output (batch_size, target)
                    # print ("data: ", np.mean(x0), np.mean(x1), np.mean(y0))
                    # print (x0) 
                    # print ("x0 shape: ", x0.shape)
                    # print ("y0 shape: ", y0.shape)
                    if (("train_LSTM_FD" in self._settings)
                        and (self._settings["train_LSTM_FD"] == True)):
                        score = self._model._forward_dynamics_net.fit([x0, x1], [y0],
                                  epochs=1, 
                                  batch_size=sequences0.shape[0],
                                  verbose=0
                                  )
                        # print ("lstm train loss: ", score.history['loss'])
                        loss_.append(np.mean(score.history['loss']))
                    if (("train_LSTM_Reward" in self._settings)
                        and (self._settings["train_LSTM_Reward"] == True)):  
                        score = self._model._reward_net.fit([x0, x1], [y0],
                                  epochs=1, 
                                  batch_size=sequences0.shape[0],
                                  verbose=0
                                  )
                        # print ("lstm train loss: ", score.history['loss'])
                        loss_.append(np.mean(score.history['loss']))
            else:
                # print ("targets_[:,:,0]: ", np.mean(targets_, axis=1))
                targets__ = np.mean(targets_, axis=1)
                if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
                    score = self._model._forward_dynamics_net.fit([sequences0, sequences1], [targets__],
                                  epochs=1, 
                                  batch_size=sequences0.shape[0],
                                  verbose=0
                                  )
                    loss_.append(np.mean(score.history['loss']))
                    
                if (("train_LSTM_Reward" in self._settings)
                    and (self._settings["train_LSTM_Reward"] == True)):
                    
                    if (("train_lstm_fd_and_reward_together" in self._settings)
                        and (self._settings["train_lstm_fd_and_reward_together"] == True)):
                        score = self._model._reward_net.fit([sequences0, sequences1], [targets__, targets_],
                                      epochs=1, 
                                      batch_size=sequences0.shape[0],
                                      verbose=0
                                      )
                    else:
                        score = self._model._reward_net.fit([sequences0, sequences1], [targets__],
                                      epochs=1, 
                                      batch_size=sequences0.shape[0],
                                      verbose=0
                                      )
                    loss_.append(np.mean(score.history['loss']))
            
            return np.mean(loss_)
        else:
            te_pair1, te_pair2, te_y = create_pairs2(states_, self._settings)
        self._updates += 1
        if (batch_size is None):
            batch_size_=states.shape[0]
        else:
            batch_size_=batch_size
        loss = 0
        # dist_ = np.array(self._contrastive_loss([te_pair1, te_pair2, 0]))[0]
        # dist = np.mean(dist_)
        te_y = np.array(te_y)
        # print("Distance: ", dist)
        # print("targets: ", te_y)
        # print("pairs: ", te_pair1)
        # print("Distance.shape, targets.shape: ", dist_.shape, te_y.shape)
        # print("Distance, targets: ", np.concatenate((dist_, te_y), axis=1))
        # if ( dist > 0):
        score = self._model._forward_dynamics_net.fit([te_pair1, te_pair2], te_y,
          epochs=updates, batch_size=batch_size_,
          verbose=0,
          shuffle=True
          )
        loss = np.mean(score.history['loss'])
            # print ("loss: ", loss)
        return loss
    
    def predict_encoding(self, state):
        """
            Compute distance between two states
        """
        # state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
            h_a = self._model.processed_a.predict([np.array([state])])
        else:
            h_a = self._model._forward_dynamics_net.predict([state])[0]
        return h_a
    
    def predict(self, state, state2):
        """
            Compute distance between two states
        """
        # print("state shape: ", np.array(state).shape)
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        state2 = np.array(norm_state(state2, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if ((("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True))
                    # or
                    # settings["use_learned_reward_function"] == "dual"
                    ):
            ### Used because we need to keep two separate RNN networks and not mix the hidden states
            h_a = self._model.processed_a.predict([np.array([state])])
            h_b = self._model.processed_b.predict([np.array([state2])])
            state_ = self._distance_func_np((h_a, h_b))[0]
            # print ("siamese dist: ", state_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
        # dist_ = np.array(self._contrastive_loss([te_pair1, te_pair2, 0]))[0]
        # print("state_ shape: ", np.array(state_).shape)
        return state_
    
    def predictWithDropout(self, state, action):
        # "dropout"
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        state_ = scale_state(self._forwardDynamics_drop()[0], self.getStateBounds())
        return state_
    
    def predict_std(self, state, action, p=1.0):
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        state_ = self._forwardDynamics_std() * (action_bound_std(self.getStateBounds()))
        return state_
    
    def predict_reward(self, state, state2):
        """
            Predict reward which is inverse of distance metric
        """
        # print ("state bounds length: ", self.getStateBounds())
        # print ("fd: ", self)
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        state2 = np.array(norm_state(state2, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_Reward" in self._settings)
            and (self._settings["train_LSTM_Reward"] == True)):
            ### Used because we need to keep two separate RNN networks and not mix the hidden states
            # print ("State shape: ", np.array([np.array([state])]).shape)
            h_a = self._model.processed_a_r.predict([np.array([state])])
            h_b = self._model.processed_b_r.predict([np.array([state2])])
            reward_ = self._distance_func_np((h_a, h_b))[0]
            # print ("siamese dist: ", state_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            predicted_reward = self._model._reward_net.predict([state, state2])[0]
            # reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
            reward_ = predicted_reward
            
        return reward_
    
    def predict_reward_encoding(self, state):
        """
            Predict reward which is inverse of distance metric
        """
        # state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_Reward" in self._settings)
            and (self._settings["train_LSTM_Reward"] == True)):
            h_a = self._model.processed_a_r.predict([np.array([state])])
        else:
            h_a = self._model._reward_net.predict([state])[0]
            # reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
            
        return h_a
    
    def predict_batch(self, states, actions):
        ## These input should already be normalized.
        return self.fd([states, actions, 0])[0]
    
    def predict_reward_batch(self, states, actions):
        """
            This data should already be normalized
        """
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        predicted_reward = self.reward([states, actions, 0])[0]
        return predicted_reward
    
    def predict_reward_(self, states, states2):
        """
            This data should already be normalized
        """
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        states = np.array(norm_state(states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        actions = np.array(norm_state(states2, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        h_a = self._model.processed_a_r_seq.predict([states])
        h_b = self._model.processed_b_r_seq.predict([states2])
        # print ("h_b shape: ", h_b.shape) 
        predicted_reward = np.array([self._distance_func_np((np.array([h_a_]), np.array([h_b_])))[0] for h_a_, h_b_ in zip(h_a[0], h_b[0])])
        # print ("predicted_reward_: ", predicted_reward)
        # predicted_reward = self._model._reward_net_seq.predict([states, actions], batch_size=1)[0]
        return predicted_reward

    def bellman_error(self, states, actions, result_states, rewards):
        self.reset()
        if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
            sequences0, sequences1, targets_ = create_sequences(states, result_states, self._settings)
            sequences0 = np.array(sequences0)
            sequences1 = np.array(sequences1)
            targets_ = np.array(targets_)
            errors=[]
            if ("train_LSTM_FD_stateful" in self._settings
                and (self._settings["train_LSTM_FD_stateful"] == True)
                # and False
                ):
                for k in range(sequences0.shape[1]):
                    ### shaping data
                    # print (k)
                    x0 = np.array(sequences0[:,[k]])
                    x1 = np.array(sequences1[:,[k]])
                    y0 = np.array(targets_[:,k]) ### For now reduce the dimensionality of the target because my nets output (batch_size, target)
                    predicted_y = self._model._forward_dynamics_net.predict([x0, x1], batch_size=x0.shape[0])
                    errors.append( compute_accuracy(predicted_y, y0) )
            else:
                predicted_y = self._model._forward_dynamics_net.predict([sequences0, sequences1], batch_size=sequences0.shape[0])
                # print ("fd error, predicted_y: ", predicted_y)
                targets__ = np.mean(targets_, axis=1)
                # print ("fd error, targets_ : ", targets_)
                # print ("fd error, targets__: ", targets__)
                errors.append( compute_accuracy(predicted_y, targets__) )
            # predicted_y = self._model._forward_dynamics_net.predict([np.array([[sequences0[0]]]), np.array([[sequences1[0]]])])
            # te_acc = compute_accuracy(predicted_y, np.array([targets_[0]]) )
            te_acc = np.mean(errors)
        else:
            states = np.concatenate((states, result_states), axis=0)
            te_pair1, te_pair2, te_y = create_pairs2(states, self._settings)
        
            # state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
            predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
            te_acc = compute_accuracy(predicted_y, te_y)
            
        # predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
        return te_acc
    
    def reward_error(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        self.reset()
        if (("train_LSTM_Reward" in self._settings)
                    and (self._settings["train_LSTM_Reward"] == True)):
            sequences0, sequences1, targets_ = create_sequences(states, result_states, self._settings)
            sequences0 = np.array(sequences0)
            sequences1 = np.array(sequences1)
            targets_ = np.array(targets_)
            errors=[]
            if ("train_LSTM_FD_stateful" in self._settings
                and (self._settings["train_LSTM_FD_stateful"] == True)
                # and False
                ):
                for k in range(sequences0.shape[1]):
                    ### shaping data
                    # print (k)
                    x0 = np.array(sequences0[:,[k]])
                    x1 = np.array(sequences1[:,[k]])
                    y0 = np.array(targets_[:,k]) ### For now reduce the dimensionality of the target because my nets output (batch_size, target)
                    predicted_y = self._model._reward_net.predict([x0, x1], batch_size=x0.shape[0])
                    errors.append( compute_accuracy(predicted_y, y0) )
            else:
                predicted_y = self._model._reward_net.predict([sequences0, sequences1], batch_size=sequences0.shape[0])
                # print ("fd error, predicted_y: ", predicted_y)
                targets__ = np.mean(targets_, axis=1)
                # print ("fd error, targets_ : ", targets_)
                # print ("fd error, targets__: ", targets__)
                errors.append( compute_accuracy(predicted_y, targets__) )
            # predicted_y = self._model._forward_dynamics_net.predict([np.array([[sequences0[0]]]), np.array([[sequences1[0]]])])
            # te_acc = compute_accuracy(predicted_y, np.array([targets_[0]]) )
            te_acc = np.mean(errors)
        else:
            states = np.concatenate((states, result_states), axis=0)
            te_pair1, te_pair2, te_y = create_pairs2(states, self._settings)
        
            # state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
            predicted_y = self._model._reward_net.predict([te_pair1, te_pair2])
            te_acc = compute_accuracy(predicted_y, te_y)
            
        # predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
        return te_acc

    def saveTo(self, fileName):
        # print(self, "saving model")
        import h5py
        hf = h5py.File(fileName+"_bounds.h5", "w")
        hf.create_dataset('_state_bounds', data=self.getStateBounds())
        hf.create_dataset('_reward_bounds', data=self.getRewardBounds())
        hf.create_dataset('_action_bounds', data=self.getActionBounds())
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("fd save self.getStateBounds(): ", len(self.getStateBounds()[0]))
        # hf.create_dataset('_resultgetStateBounds()', data=self.getResultStateBounds())
        # print ("fd: ", self)
        hf.flush()
        hf.close()
        suffix = ".h5"
        ### Save models
        # self._model._actor_train.save(fileName+"_actor_train"+suffix, overwrite=True)
        self._model._forward_dynamics_net.save(fileName+"_FD"+suffix, overwrite=True)
        self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
        # print ("self._model._actor_train: ", self._model._actor_train)
        try:
            from keras.utils import plot_model
            ### Save model design as image
            plot_model(self._model._forward_dynamics_net, to_file=fileName+"_FD"+'.svg', show_shapes=True)
            plot_model(self._model._reward_net, to_file=fileName+"_reward"+'.svg', show_shapes=True)
        except Exception as inst:
            ### Maybe the needed libraries are not available
            print ("Error saving diagrams for rl models.")
            print (inst)
        
    def loadFrom(self, fileName):
        import h5py
        from util.utils import load_keras_model
        suffix = ".h5"
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Loading agent: ", fileName)
        # with K.get_session().graph.as_default() as g:
        ### Need to lead the model this way because the learning model's State expects batches...
        forward_dynamics_net = load_keras_model(fileName+"_FD"+suffix, custom_objects={'contrastive_loss': contrastive_loss})
        reward_net = load_keras_model(fileName+"_reward"+suffix, custom_objects={'contrastive_loss': contrastive_loss})
        # if ("simulation_model" in self.getSettings() and
        #     (self.getSettings()["simulation_model"] == True)):
        if (True): ### Because the simulation and learning use different model types (statefull vs stateless lstms...)
            self._model._forward_dynamics_net.set_weights(forward_dynamics_net.get_weights())
            self._model._forward_dynamics_net.optimizer = forward_dynamics_net.optimizer
            self._model._reward_net.set_weights(reward_net.get_weights())
            self._model._reward_net.optimizer = reward_net.optimizer
        else:
            self._model._forward_dynamics_net = forward_dynamics_net
            self._model._reward_net = reward_net
            
        self._forward_dynamics_net = self._model._forward_dynamics_net
        self._reward_net = self._model._reward_net
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("******** self._forward_dynamics_net: ", self._forward_dynamics_net)
        if (self._modelTarget is not None):
            self._modelTarget._forward_dynamics_net = load_keras_model(fileName+"_actor_T"+suffix)
            self._modelTarget._reward_net = load_keras_model(fileName+"_reward_net_T"+suffix)
        # self._model._actor_train = load_keras_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("fd load self.getStateBounds(): ", len(self.getStateBounds()[0]))
        # self._resultgetStateBounds() = np.array(hf.get('_resultgetStateBounds()'))
        hf.close()
        