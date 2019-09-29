import numpy as np
import sys
from model.ModelUtil import *
from model.LearningUtil import loglikelihood, loglikelihoodMEAN, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat, likelihood, loglikelihoodMEAN
from model.LearningUtil import loglikelihood, likelihood, likelihoodMEAN, kl, kl_D, entropy, flatgrad, zipsame, get_params_flat, setFromFlat
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras
from keras.models import Sequential, Model
from util.SimulationUtil import createForwardDynamicsNetwork
from algorithm.SiameseNetwork import *
from util.utils import rlPrint

# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.KERASAlgorithm import KERASAlgorithm


def create_sequences(traj0, traj1, settings):
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
    noise_scale = 0.03
    noise_scale_dense = 0.001
    target_noise_scale = 0.05
    sequences0 = []
    sequences1 = []
    targets_ = []
    for tr0, tr1 in zip(traj0, traj1): ### for each trajectory pair
        ### transform data into poses
        # print ("tr0 shape: ", tr0.shape)
        # print ("tr1 shape: ", tr1.shape)
        trajPoses0 = tr0[:, :settings["dense_state_size"]]
        trajPoses1 = tr1[:, :settings["dense_state_size"]]
        tar_shape = (len(tr0)-1, 1)
        if (len(tr0) == 1):
            tar_shape = (len(tr0), 1)
            ### Same data with a little noise
            sequences0.append(tr0 + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))
            sequences1.append(trajPoses0 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            sequences0.append(tr0 + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))
            sequences1.append(trajPoses0 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            sequences0.append(tr1 + np.random.normal(loc=0, scale=noise_scale, size=tr1.shape))
            sequences1.append(trajPoses1 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            sequences0.append(tr1 + np.random.normal(loc=0, scale=noise_scale, size=tr1.shape))
            sequences1.append(trajPoses1 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            ### Probably different (if not from beginning of epoch) data with a little noise
            sequences0.append(tr0 + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))
            sequences1.append(trajPoses1 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))
            targets = np.zeros(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            sequences0.append(tr0 + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))
            sequences1.append(trajPoses1 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))
            targets = np.zeros(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            sequences0.append(tr1 + np.random.normal(loc=0, scale=noise_scale, size=tr1.shape))
            sequences1.append(trajPoses0 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))
            targets = np.zeros(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            sequences0.append(tr1 + np.random.normal(loc=0, scale=noise_scale, size=tr1.shape))
            sequences1.append(trajPoses0 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))
            targets = np.zeros(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            
        elif (len(tr0) == 2):
            
            tar_shape = (len(tr0), 1)
            ### Same clips
            sequences0.append(tr0 + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))
            sequences1.append(trajPoses0 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            sequences0.append(tr0 + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))
            sequences1.append(trajPoses0 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            sequences0.append(tr1 + np.random.normal(loc=0, scale=noise_scale, size=tr1.shape))
            sequences1.append(trajPoses1 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            sequences0.append(tr1 + np.random.normal(loc=0, scale=noise_scale, size=tr1.shape))
            sequences1.append(trajPoses1 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            ### clips with repeated frames
            sequences0.append([tr0[0]] + tr0[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))
            sequences1.append([trajPoses0[0]] + trajPoses0[1:] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            sequences0.append([tr1[0]] + tr1[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr1.shape))
            sequences1.append([trajPoses1[0]] + trajPoses1[1:] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            ### reversed versions of the same trajectories
            sequences0.append(list(reversed(tr0 + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))))
            sequences1.append(trajPoses0 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))
            targets = np.zeros(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            sequences0.append(tr0 + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))
            sequences1.append(list(reversed(trajPoses0 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))))
            targets = np.zeros(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            sequences0.append(list(reversed(tr1 + np.random.normal(loc=0, scale=noise_scale, size=tr1.shape))))
            sequences1.append(trajPoses1 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))
            targets = np.zeros(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            sequences0.append(tr1 + np.random.normal(loc=0, scale=noise_scale, size=tr1.shape))
            sequences1.append(list(reversed(trajPoses1 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))))
            targets = np.zeros(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            
            # if ("include_agent_imitator_pairs" in settings
            #     and (settings["include_agent_imitator_pairs"] == True)):
            ### Versions of two different trajectories
            if (np.random.rand() > 0.5):
                sequences0.append(tr0 + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))
                sequences1.append(trajPoses1 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
                
                sequences0.append(tr1 + np.random.normal(loc=0, scale=noise_scale, size=tr1.shape))
                sequences1.append(trajPoses0 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            else:
                ### Comparing agent sequence and imitation char sequence
                sequences0.append(list(reversed(tr0 + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))))
                sequences1.append(list(reversed(trajPoses1 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
                
                sequences0.append(list(reversed(tr1 + np.random.normal(loc=0, scale=noise_scale, size=tr0.shape))))
                sequences1.append(list(reversed(trajPoses0 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
                 
            # print ("sequences0: ", np.array(sequences0).shape)
            
        else:
            ### basic for now
            
            ### Noisy versions of the same trajectories
            sequences0.append(tr0[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
            sequences1.append(trajPoses0[1:] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            sequences0.append(tr1[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
            sequences1.append(trajPoses1[1:] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            sequences0.append(tr0[:-1] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
            sequences1.append(trajPoses0[:-1] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            sequences0.append(tr1[:-1] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
            sequences1.append(trajPoses1[:-1] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
            targets = np.ones(tar_shape)
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            
            ### Out of sync versions of the same trajectories
            sequences0.append(tr0[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
            sequences1.append(np.concatenate(([trajPoses0[2]], trajPoses0[2:]), axis=0) + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))
            targets = np.ones(tar_shape) - compare_adjustment
            targets[0] = 0
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            # 
            sequences0.append(tr0[:-1] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
            sequences1.append(np.concatenate((trajPoses0[:-2], [trajPoses0[-2]]), axis=0) + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))
            targets = np.ones(tar_shape) - compare_adjustment
            targets[-1] = 0
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            ### Out of sync versions of the same trajectories
            sequences0.append(tr1[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
            sequences1.append(np.concatenate(([trajPoses1[2]], trajPoses1[2:]), axis=0) + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
            targets = np.ones(tar_shape) - compare_adjustment
            targets[0] = 0
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            # 
            sequences0.append(tr1[:-1] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
            sequences1.append(np.concatenate((trajPoses1[:-2], [trajPoses1[-2]]), axis=0) + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
            targets = np.ones(tar_shape) - compare_adjustment
            targets[-1] = 0
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            ### reversed versions of the same trajectories
            if (np.random.rand() > 0.5):
                sequences0.append(list(reversed(tr0[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))))
                sequences1.append(trajPoses0[:-1] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            else:
                sequences0.append(tr0[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append(list(reversed(trajPoses0[1:] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            if (np.random.rand() > 0.5):
                sequences0.append(list(reversed(tr1[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))))
                sequences1.append(trajPoses1[:-1] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            else:
                sequences0.append(tr1[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
                sequences1.append(list(reversed(trajPoses1[1:] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            ### Random frozen frame versions of sequences
            if (np.random.rand() > 0.5):
                sequences0.append(tr1[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
                sequences1.append([trajPoses1[np.random.choice(range(len(tr1)))] * len(tr1[1:])] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            else:
                sequences0.append(tr1[:-1] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
                sequences1.append([trajPoses1[np.random.choice(range(len(tr1)))] * len(tr1[1:])] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            if (np.random.rand() > 0.5):
                sequences0.append([tr1[np.random.choice(range(len(tr1)))] * len(tr1[1:])] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
                sequences1.append(trajPoses1[1:] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            else:
                sequences0.append([tr1[np.random.choice(range(len(tr1)))] * len(tr1[1:])] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
                sequences1.append(trajPoses1[:-1] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[:-1].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            if (np.random.rand() > 0.5):
                sequences0.append(tr0[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append([trajPoses0[np.random.choice(range(len(trajPoses0)))] * len(trajPoses0[1:])] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            else:
                sequences0.append(tr0[:-1] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append([trajPoses0[np.random.choice(range(len(trajPoses0)))] * len(tr1[1:])] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            if (np.random.rand() > 0.5):
                sequences0.append([tr0[np.random.choice(range(len(tr0)))] * len(tr0[1:])] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append(trajPoses0[1:] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            else:
                sequences0.append([tr0[np.random.choice(range(len(tr0)))] * len(tr0[1:])] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append(trajPoses0[:-1] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[:-1].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            ### Randomly shuffled sequences
            indicies = range(len(tr1))
            # print ("indicies: ", indicies)
            # print ("choice: ", np.random.choice(indicies, len(tr1[1:])))
            if (np.random.rand() > 0.5):
                sequences0.append(tr0[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append(np.array(trajPoses0)[np.random.choice(indicies, len(trajPoses0[1:]))] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            else:
                sequences0.append(tr1[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
                sequences1.append(np.array(trajPoses1)[np.random.choice(indicies, len(trajPoses1[1:]))] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            if (np.random.rand() > 0.5):
                sequences0.append(np.array(tr0)[np.random.choice(indicies, len(tr0[1:]))] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append(trajPoses0[1:] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            else:
                sequences0.append(np.array(tr1)[np.random.choice(indicies, len(tr1[1:]))] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
                sequences1.append(trajPoses1[1:] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            ## Both random sequences
            if (np.random.rand() > 0.5):
                sequences0.append(np.array(tr0)[np.random.choice(indicies, len(tr0[1:]))] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append(np.array(trajPoses0)[np.random.choice(indicies, len(trajPoses0[1:]))] + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0[1:].shape))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            #
            else:
                sequences0.append(np.array(tr1)[np.random.choice(indicies, len(tr1[1:]))] + np.random.normal(loc=0, scale=noise_scale, size=tr1[1:].shape))
                sequences1.append(np.array(trajPoses1)[np.random.choice(indicies, len(trajPoses1[1:]))] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
                targets = np.zeros(tar_shape)
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            
            if ("include_agent_imitator_pairs" in settings
                and (settings["include_agent_imitator_pairs"] == True)):
                ### Versions of two different trajectories
                sequences0.append(tr0[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append(trajPoses1[1:] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
                
                sequences0.append(tr0[:-1] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append(trajPoses1[:-1] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
                
                
                ### More Out of sync versions of two different trajectories
                sequences0.append(tr0[1:] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append(trajPoses1[:-1] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
                
                sequences0.append(tr0[:-1] + np.random.normal(loc=0, scale=noise_scale, size=tr0[1:].shape))
                sequences1.append(trajPoses1[1:] + np.random.normal(loc=0, scale=noise_scale, size=trajPoses1[1:].shape))
                targets = np.zeros(tar_shape) + compare_adjustment
                targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))

        # print ("Created trajectories: ", len(targets_))
    
    return sequences0, sequences1, targets_

def create_multitask_sequences(traj0, traj1, task_ids, settings):
    '''Positive and negative sequence creation.
    Alternates between positive and negative pairs.
    produces N sequences from two
    
    class ids are stored in task_ids
    '''
    ### Transform data into poses
    noise_scale = 0.03
    noise_scale_dense = 0.001
    target_noise_scale = 0.05
    compare_adjustment = 0.0
    if ("imperfect_compare_offset" in settings):
        compare_adjustment = settings["imperfect_compare_offset"]
    sequences0 = []
    sequences1 = []
    targets_ = []
    for i in range(len(traj0)):
    # for tr0, task_tr0 in zip(traj0, task_ids): ### for each trajectory pair
        tar_shape = (len(traj0[i]), 1)
        
        for j in range(len(traj0)):
        # for tr1, task_tr1 in zip(traj0, task_ids): ### for each trajectory pair
            trajPoses0 = traj0[i][:, :settings["dense_state_size"]]
            trajPoses1 = traj1[j][:, :settings["dense_state_size"]]
            ### Noisy versions of the same trajectories
            sequences0.append(traj0[i] + np.random.normal(loc=0, scale=noise_scale, size=traj0[i].shape))
            sequences1.append(trajPoses0 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses0.shape))
            
            sequences0.append(traj1[i] + np.random.normal(loc=0, scale=noise_scale, size=traj1[i].shape))
            sequences1.append(trajPoses1 + np.random.normal(loc=0, scale=noise_scale_dense, size=trajPoses1.shape))
            # print ("task_tr0[0][0] == task_tr1[0][0]", task_tr0[0][0], " == ", task_tr1[0][0])
            # print ("settings['worker_to_task_mapping'][task_tr0[0]] == settings['worker_to_task_mapping'][task_tr1[0]]", 
            #        settings["worker_to_task_mapping"][task_tr0[0][0]]," == ", settings["worker_to_task_mapping"][task_tr1[0][0]])
            if (settings["worker_to_task_mapping"][task_ids[i][0][0]] == settings["worker_to_task_mapping"][task_ids[j][0][0]]): ### same task
                if ( i == j ): ### same trajectory
                    targets = np.ones(tar_shape)
                else: ### Same class
                    targets = np.ones(tar_shape) - compare_adjustment
            else:
                targets = np.zeros(tar_shape)
            # print ("targets", targets)
            # print ("task_ids[i][0][0]: ", task_ids[i][0][0], " task_ids[j][0][0]: ", task_ids[j][0][0])
            # print ("multitask targets", np.mean(targets))
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
            targets_.append(np.clip(targets + np.random.normal(loc=0, scale=target_noise_scale, size=tar_shape), 0.01, 0.98))
        
        
    
    return sequences0, sequences1, targets_
        
def create_pairs2(x, settings):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.\
    
    :Input x
        Should be a array of [[image:pose], ... , [image:pose]]
        
    :Return (images, poses, labels)
         
    '''
    poses = x[:, :settings["dense_state_size"]]
    # print ("imgs shape: ", np.array(imgs).shape)
    # print ("poses shape: ", np.array(poses).shape)
    
    target_noise_scale = 0.05
    compare_adjustment = 0.0
    if ("imperfect_compare_offset" in settings):
        compare_adjustment = settings["imperfect_compare_offset"]
    noise_scale = 0.02
    pair1 = []
    pair2 = []
    labels = []
    n = x.shape[0] - 1
    indices = list(np.random.randint(low=0, high=n, size=n))
    for i in range(n):
        ### Identical pair
        i = indices[i]
        noise = np.random.normal(loc=0, scale=noise_scale, size=x[i].shape)
        x1 = [x[i] + noise]

        pair1 += x1
        pair2 += [poses[i] + np.random.normal(loc=0, scale=noise_scale/2.0, size=poses[i].shape)]
        ### Different pair
        z=i
        while (z == i): ## get result that is not the same
            z = np.random.randint(low=0, high=n)
        noise = np.random.normal(loc=0, scale=noise_scale, size=x[i].shape)
        x1 = [x[i] + noise]

        pair1 += x1
        pair2 += [poses[z] + np.random.normal(loc=0, scale=noise_scale/2.0, size=poses[z].shape)]

        labels += [np.clip(1 + np.random.normal(loc=0, scale=target_noise_scale, size=1), 0.01, 0.98),
                    np.clip(0 + np.random.normal(loc=0, scale=target_noise_scale, size=1), 0.01, 0.98)]
    # print (np.array(pair1).shape, np.array(pair2).shape, np.array(labels).shape)
    return np.array(pair1), np.array(pair2), np.array(labels)

class MultiModalSiameseNetwork(KERASAlgorithm):
    """
         This method uses two different types of data and learns a distance function between them.
         In this case the first type of data is pixles and the second is dense pose data.
         
         Notes:
         Maybe I can just let the first model ignore the pose features, i.e. not merge them...
    """
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        super(MultiModalSiameseNetwork,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        self._model = model
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-6
        
        condition_reward_on_result_state = False
        self._train_combined_loss = False
        
        ### Need to create a new model that uses a different network
        settings__ = copy.deepcopy(self.getSettings())
        settings__["fd_network_layer_sizes"] = settings__["fd2_network_layer_sizes"]
        settings__["fd_num_terrain_features"] = 0
        if ("remove_character_state_features" in settings__):
            settings__["state_bounds"][0] = settings__["state_bounds"][0][:-settings__["remove_character_state_features"]]
            settings__["state_bounds"][1] = settings__["state_bounds"][1][:-settings__["remove_character_state_features"]]
        print ("****** Creating dense pose encoding network")
        self._model2 = createForwardDynamicsNetwork(settings__["state_bounds"], 
                                                         settings__["action_bounds"], settings__,
                                                         stateName="State_", resultStateName="ResultState_")

        self._inputs_a = self._model.getStateSymbolicVariable()
        self._inputs_b = self._model2.getStateSymbolicVariable() 
        self._model._forward_dynamics_net = Model(inputs=[self._inputs_a], outputs=self._model._forward_dynamics_net)
        self._model2._forward_dynamics_net = Model(inputs=[self._inputs_b], outputs=self._model2._forward_dynamics_net)
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("FD Conv Net summary: ", self._model._forward_dynamics_net.summary())
                print("FD Target Net summary: ", self._model2._forward_dynamics_net.summary())
        
        if ("train_LSTM_Reward" in self.getSettings()
            and (self.getSettings()["train_LSTM_Reward"] == True)):
            self._inputs_aa = self._model.getResultStateSymbolicVariable()
            self._inputs_bb = self._model2.getResultStateSymbolicVariable()
        else:
            self._inputs_aa = self._model.getStateSymbolicVariable()
            self._inputs_bb = self._model2.getStateSymbolicVariable()
        print ("self._model._State_: ", repr(self._model._State_))
        self._model._reward_net = Model(inputs=[self._model._State_], outputs=self._model._reward_net)
        self._model2._reward_net = Model(inputs=[self._model2._State_], outputs=self._model2._reward_net)
        if (print_info):
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("FD Reward Net summary: ", self._model._reward_net.summary())
                print("FD Target Reward Net summary: ", self._model2._reward_net.summary())
                

        MultiModalSiameseNetwork.compile(self)
    
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        
        print ("*** self._model.getStateSymbolicVariable() shape: ", repr(keras.backend.int_shape(self._model.getStateSymbolicVariable())))
        print ("*** self._model.getResultStateSymbolicVariable() shape: ", repr(keras.backend.int_shape(self._model.getResultStateSymbolicVariable())))
        print ("*** self._model2.getStateSymbolicVariable() shape: ", repr(keras.backend.int_shape(self._model2.getStateSymbolicVariable())))
        print ("*** self._model2.getResultStateSymbolicVariable() shape: ", repr(keras.backend.int_shape(self._model2.getResultStateSymbolicVariable())))
        state_copy = keras.layers.Input(shape=keras.backend.int_shape(self._inputs_b)[1:], name="State_2")
        result_state_copy = keras.layers.Input(shape=keras.backend.int_shape(self._inputs_bb)[1:]
                                                                              , name="ResultState_2"
                                                                              )
        processed_a = self._model._forward_dynamics_net(self._inputs_a)
        print ("processed_a shape: ", repr(processed_a))
        self._model.processed_a = Model(inputs=[self._inputs_a], outputs=processed_a)
        processed_b = self._model2._forward_dynamics_net(state_copy)
        print ("processed_a shape: ", repr(processed_b))
        self._model.processed_b = Model(inputs=[state_copy], outputs=processed_b)
        ### Convert sequence input to sequence output
        network_ = keras.layers.TimeDistributed(self._model.processed_a, input_shape=(None, 1, self._state_length))(self._model.getResultStateSymbolicVariable())
        print ("network_: ", repr(network_))
        network_b = keras.layers.TimeDistributed(self._model.processed_b, input_shape=(None, 1, self._state_length))(self._model2.getResultStateSymbolicVariable())
        print ("network_b: ", repr(network_b))
        processed_a_r = self._model._reward_net(network_)
        self._model.processed_a_r = Model(inputs=[self._inputs_aa], outputs=processed_a_r)
        use_same_rnn_net = False
        if (use_same_rnn_net):
            processed_b_r = self._model._reward_net(network_b)
            self._model.processed_b_r = Model(inputs=[self._inputs_bb], outputs=processed_b_r)
        else:
            processed_b_r = self._model2._reward_net(network_b)
            self._model.processed_b_r = Model(inputs=[self._inputs_bb], outputs=processed_b_r)
            
        self._model.processed_a_r_target = keras.models.clone_model(self._model.processed_a_r)
        self._model.processed_b_r_target = keras.models.clone_model(self._model.processed_b_r)
        
        distance_fd = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        distance_r = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a_r, processed_b_r])

        self._model._forward_dynamics_net = Model(inputs=[self._inputs_a
                                                          ,state_copy
                                                          ]
                                                  , outputs=distance_fd
                                                  )
        
        self._model._reward_net = Model(inputs=[self._inputs_aa
                                              ,self._inputs_bb
                                              ]
                                              , outputs=distance_r
                                              )
        
        rnn_encoding_ = keras.layers.Input(shape=keras.backend.int_shape(processed_a_r)[1:]
                                                                              , name="RNN_Encoding_"
                                                                              )
        print ("*** LSTM encoding shape shape: ", repr(rnn_encoding_))
        # sys.exit()
        distance_r2 = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a_r, rnn_encoding_])
        self._model._reward_net2 = Model(inputs=[self._inputs_aa
                                              ,rnn_encoding_
                                              ]
                                              , outputs=distance_r2
                                              )
        
        distance_r3 = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_b_r, rnn_encoding_])
        self._model._reward_net3 = Model(inputs=[self._inputs_bb
                                      ,rnn_encoding_
                                      ]
                                      , outputs=distance_r3
                                      )

        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=2.5)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
        self._model._forward_dynamics_net.compile(loss="mse", optimizer=sgd)

        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=1.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
        self._model._reward_net.compile(loss=contrastive_loss, optimizer=sgd)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=1.0)
        self._model._reward_net2.compile(loss=contrastive_loss, optimizer=sgd)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=1.0)
        self._model._reward_net3.compile(loss=contrastive_loss, optimizer=sgd)
        
        self._contrastive_loss = K.function([self._inputs_a, 
                                             self._inputs_b,
                                             K.learning_phase()], 
                                            [distance_fd])
        
        self._contrastive_loss_r = K.function([self._inputs_aa, 
                                             self._inputs_bb,
                                             K.learning_phase()], 
                                            [distance_r])
        # self.reward = K.function([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable(), K.learning_phase()], [self._reward])
        
    def reset(self):
        """
            Reset any state for the agent model
        """
        self._model.reset()
        self._model._reward_net.reset_states()
        self._model._reward_net2.reset_states()
        self._model._reward_net3.reset_states()
        self._model._forward_dynamics_net.reset_states()
        if not (self._model2 is None):
            self._model2.reset()
            
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model._forward_dynamics_net.get_weights()))
        params.append(copy.deepcopy(self._model._reward_net.get_weights()))
        params.append(copy.deepcopy(self._model2._forward_dynamics_net.get_weights()))
        params.append(copy.deepcopy(self._model2._reward_net.get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        self._model._forward_dynamics_net.set_weights(params[0])
        self._model._reward_net.set_weights(params[1])
        self._model2._forward_dynamics_net.set_weights(params[2])
        self._model2._reward_net.set_weights(params[3])
        
    def setGradTarget(self, grad):
        self._fd_grad_target_shared.set_value(grad)
        
    def getGrads(self, states, actions, result_states, v_grad=None, alreadyNormed=False):
        if ( alreadyNormed == False ):
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            result_states = np.array(norm_state(result_states, self._state_bounds), dtype=self.getSettings()['float_type'])
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
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            # rewards = np.array(norm_state(rewards, self._reward_bounds), dtype=self.getSettings()['float_type'])
        # self.setData(states, actions)
        return self._get_grad_reward([states, actions, 0])[0]
    
    def updateTargetModel(self):
        rlPrint(self.getSettings(), 'train', "Updating fd target model")
        ### current parameters should be different.
        assert not np.allclose(self._model.processed_a_r.get_weights()[0], self._model.processed_a_r_target.get_weights()[0])
        self._model.processed_a_r_target.set_weights(self._model.processed_a_r.get_weights())
        self._model.processed_b_r_target.set_weights(self._model.processed_b_r_target.get_weights())
        assert np.allclose(self._model.processed_a_r.get_weights()[0], self._model.processed_a_r_target.get_weights()[0])
        
                
    def train(self, states, actions, result_states, rewards, falls=None, updates=1, batch_size=None, p=1, lstm=True, datas=None):
        """
            states will come for the agent and
            results_states can come from the imitation agent
        """
        self.reset()
        if ('anneal_learning_rate' in self.getSettings()
            and (self.getSettings()['anneal_learning_rate'] == True)):
            K.set_value(self._model._forward_dynamics_net.optimizer.lr, np.float32(self.getSettings()['fd_learning_rate']) * p)
        if ("replace_next_state_with_imitation_viz_state" in self.getSettings()
            and (self.getSettings()["replace_next_state_with_imitation_viz_state"] == True)):
            states_ = np.concatenate((states, result_states), axis=0)
        if (((("train_LSTM_FD" in self._settings)
                and (self._settings["train_LSTM_FD"] == True))
            or
            (("train_LSTM_Reward" in self._settings)
                and (self._settings["train_LSTM_Reward"] == True))
            ) 
            and lstm):
            ### result states can be from the imitation agent.
            if (falls is None):
                sequences0, sequences1, targets_ = create_sequences(states, result_states, self._settings)
            else:
                sequences0, sequences1, targets_ = create_multitask_sequences(states, result_states, datas["task_id"], self._settings)
            """
            for jk in range(len(sequences0)):
                print ("sequences0 " , jk, ": len = ", len(sequences0[jk]))
            """
            # print ("sequences0 shape: ", np.array(sequences0).shape)
            sequences0 = np.array(sequences0)
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
                # print ("sequences0 shape: ", np.array(sequences0).shape)
                if ("remove_character_state_features" in self._settings):
                    ### Remove ground reaction forces from state
                    sequences1 = sequences1[:, :, :-self._settings["remove_character_state_features"]]
                if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
                    score = self._model._forward_dynamics_net.fit([sequences0, sequences1], [targets__],
                                  epochs=1, 
                                  batch_size=sequences0.shape[0],
                                  verbose=0
                                  )
                    loss_.append(np.mean(score.history['loss']))
                    print ("lstm fd training")
                    
                if (("train_LSTM_Reward" in self._settings)
                    and (self._settings["train_LSTM_Reward"] == True)):
                    """
                    sequences0_ = []
                    sequences1_ = []
                    for m in range(len(sequences0[0])):
                        sequences0_.extend(sequences0[:,m,:])
                        sequences1_.extend(sequences1[:,m,:])
                    sequences0_ = np.array(sequences0_)
                    sequences1_ = np.array(sequences1_)
                    score_ = self._model._forward_dynamics_net.fit([sequences0_, sequences1_], [np.zeros((sequences0_.shape[0],1))],
                                      epochs=len(sequences0[0]), 
                                      batch_size=sequences0.shape[0],
                                      verbose=0
                                      )
                        # print ("score_: ", score_)
                    """
                    # if (np.random.rand() > 0.5):
                    h_a = self._model.processed_b_r_target.predict([sequences1])
                    score = self._model._reward_net2.fit([sequences0, h_a], [targets__],
                                  epochs=1, 
                                  batch_size=sequences0.shape[0],
                                  verbose=0
                                  )
                    # else:
                    h_a = self._model.processed_a_r_target.predict([sequences0])
                    # score = self._model._reward_net3.fit([sequences1, (h_a * 0.0)], [(targets__ * 0) + 1],
                    score = self._model._reward_net3.fit([sequences1, h_a], [targets__],
                                  epochs=1, 
                                  batch_size=sequences0.shape[0],
                                  verbose=0
                                  )
                    """
                    score = self._model._reward_net.fit([sequences0, sequences1], [targets__],
                                  epochs=1, 
                                  batch_size=sequences0.shape[0],
                                  verbose=0
                                  )
                    """
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
        # state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
            h_a = self._model.processed_a.predict([np.array([state])])
        else:
            h_a = self._model._forward_dynamics_net.predict([state])[0]
        return h_a
    
    def predict(self, state):
        """
            Compute distance between two states
        """
        # print("state shape: ", np.array(state).shape)
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # state2 = np.array(norm_state(state2, self._state_bounds), dtype=self.getSettings()['float_type'])
        if ((("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True))
                    # or
                    # settings["use_learned_reward_function"] == "dual"
                    ):
            
            state2 = state[:, :self._settings["dense_state_size"]]
            if ("remove_character_state_features" in self._settings):
                ### Remove ground reaction forces from state
                state2 = state2[:, :-self._settings["remove_character_state_features"]]
            ### Used because we need to keep two separate RNN networks and not mix the hidden states
            h_a = self._model.processed_a.predict([np.array([state])])
            h_b = self._model.processed_b.predict([np.array([state2])])
            state_ = euclidean_distance_np((h_a, h_b))[0]
            # print ("siamese dist: ", state_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            # print ("State shape: ", state.shape, " state2 shape: ", state2.shape)
            # state2 = np.array(norm_state(state2, self._state_bounds), dtype=self.getSettings()['float_type'])
            state2 = state[:, :self._settings["dense_state_size"]]
            state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
        # dist_ = np.array(self._contrastive_loss([te_pair1, te_pair2, 0]))[0]
        # print("state_ shape: ", np.array(state_).shape)
        return state_
    
    def predictWithDropout(self, state, action):
        # "dropout"
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        state_ = scale_state(self._forwardDynamics_drop()[0], self._state_bounds)
        return state_
    
    def predict_std(self, state, action, p=1.0):
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        state_ = self._forwardDynamics_std() * (action_bound_std(self._state_bounds))
        return state_
    
    def predict_reward(self, state):
        """
            Predict reward which is inverse of distance metric
        """
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # state2 = np.array(norm_state(state2, self._state_bounds), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_Reward" in self._settings)
            and (self._settings["train_LSTM_Reward"] == True)):
            # print ("state shape: ", state.shape)
            state2 = state[:, :self._settings["dense_state_size"]]
            if ("remove_character_state_features" in self._settings):
                ### Remove ground reaction forces from state
                state2 = state2[:, :-self._settings["remove_character_state_features"]]
            ### Used because we need to keep two separate RNN networks and not mix the hidden states
            h_a = self._model.processed_a_r.predict([np.array([state])])
            h_b = self._model.processed_b_r.predict([np.array([state2])])
            reward_ = euclidean_distance_np((h_a, h_b))[0]
            # print ("siamese dist: ", state_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            state2 = state[:, :self._settings["dense_state_size"]]
            predicted_reward = self._model._reward_net.predict([state, state2])[0]
            # reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
            reward_ = predicted_reward
            
        return reward_
    
    def predict_reward_encoding(self, state):
        """
            Predict reward which is inverse of distance metric
        """
        # state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
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
                if ("remove_character_state_features" in self._settings):
                    ### Remove ground reaction forces from state
                    sequences1 = sequences1[:, :, :-self._settings["remove_character_state_features"]]
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
        
            if ("remove_character_state_features" in self._settings):
                ### Remove ground reaction forces from state
                te_pair2 = te_pair2[:, :-self._settings["remove_character_state_features"]]
            # state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
            predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
            # print ("predicted_y: ", predicted_y)
            # print ("te_y: ", te_y)
            te_acc = compute_accuracy(predicted_y, te_y)
            # te_acc = predicted_y
            # print ("te_acc: ", te_acc)
            
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
                if ("remove_character_state_features" in self._settings):
                    ### Remove ground reaction forces from state
                    sequences1 = sequences1[:, :, :-self._settings["remove_character_state_features"]]
                # predicted_y = self._model._reward_net3.predict([sequences1, np.zeros((sequences1.shape[0], 64))], batch_size=sequences0.shape[0])
                # if (np.random.rand() > 0.5):
                # h_a = self._model.processed_b_r.predict([sequences0])
                predicted_y = self._model._reward_net.predict([sequences0, sequences1], batch_size=sequences0.shape[0])
                # else:
                # h_a = self._model.processed_a_r.predict([sequences0])
                # predicted_y2 = self._model._reward_net3.predict([sequences1, h_a], batch_size=sequences0.shape[0])
                # print ("fd error, predicted_y: ", predicted_y)
                targets__ = np.mean(targets_, axis=1)
                # print ("fd error, targets_ : ", targets_)
                # print ("fd error, targets__: ", targets__)
                # errors.append( compute_accuracy(predicted_y, targets__) )
                errors.append( contrastive_loss_np(predicted_y, targets__) )
                # errors.append( predicted_y )
            # predicted_y = self._model._forward_dynamics_net.predict([np.array([[sequences0[0]]]), np.array([[sequences1[0]]])])
            # te_acc = compute_accuracy(predicted_y, np.array([targets_[0]]) )
            te_acc = np.mean(errors)
        else:
            states = np.concatenate((states, result_states), axis=0)
            te_pair1, te_pair2, te_y = create_pairs2(states, self._settings)
        
            # state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
            predicted_y = self._model._reward_net._reward_net3([te_pair1, (te_pair2 * 0.0)])
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
        # hf.create_dataset('_result_state_bounds', data=self.getResultStateBounds())
        hf.flush()
        hf.close()
        suffix = ".h5"
        ### Save models
        # self._model._actor_train.save(fileName+"_actor_train"+suffix, overwrite=True)
        self._model._forward_dynamics_net.save(fileName+"_FD"+suffix, overwrite=True)
        self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
        self._model2._forward_dynamics_net.save(fileName+"_FD_T"+suffix, overwrite=True)
        self._model2._reward_net.save(fileName+"_reward_T"+suffix, overwrite=True)
        # print ("self._model._actor_train: ", self._model._actor_train)
        
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
        if (self._model2 is not None):
            self._model2._forward_dynamics_net = load_keras_model(fileName+"_FD_T"+suffix)
            self._model2._reward_net = load_keras_model(fileName+"_reward_T"+suffix)
        # self._model._actor_train = load_keras_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("fd self.getStateBounds(): ", self.getStateBounds())
        # self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        hf.close()
        