import numpy as np
import h5py
# import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import norm_state, scale_state, norm_action, scale_action, action_bound_std, scale_reward
from algorithm.AlgorithmInterface import AlgorithmInterface
from model.LearningUtil import loglikelihood_keras, likelihood_keras, kl_keras, kl_D_keras, entropy_keras
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras
from keras.models import Sequential, Model

# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

"""
def dice_coef(y_true, y_pred, smooth, thresh):
    y_pred = y_pred > thresh
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

### Loss    
def dice_loss(smooth, thresh):
  def dice(y_true, y_pred)
    return -dice_coef(y_true, y_pred, smooth, thresh)
  return dice
"""

def flatten(data):
    
    for i in data:
        if isinstance(i, (list, tuple, np.ndarray)):
            for j in  flatten(i):
                yield j
        else:
            yield i

class KERASAlgorithm(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False):

        super(KERASAlgorithm,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False)
        
    def saveTo(self, fileName):
        # print(self, "saving model")
        import dill
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
        self._model._actor.save(fileName+"_actor"+suffix, overwrite=True)
        self._model._critic.save(fileName+"_critic"+suffix, overwrite=True)
        self._modelTarget._actor.save(fileName+"_actor_T"+suffix, overwrite=True)
        self._modelTarget._critic.save(fileName+"_critic_T"+suffix, overwrite=True)
        ### Make a temp copy of models
        # model_actor_train = self._model._actor_train
        model_actor = self._model._actor
        model_critic = self._model._critic
        modelTarget_actor = self._modelTarget._actor
        modelTarget_critic = self._modelTarget._critic
        ### Set models to none so they are not saved with this pickling... Because Keras does not pickle well.
        self._model._actor_train = None
        self._model._actor = None
        self._model._critic = None
        self._modelTarget._actor = None
        self._modelTarget._critic = None
        ### Pickle this class
        """
        suffix = ".pkl"
        file_name=fileName+suffix
        f = open(file_name, 'wb')
        dill.dump(self, f)
        f.close()
        """
        ### Restore models
        # self._model = model
        # self._modelTarget = modelTarget
        # self._model._actor_train = model_actor_train
        self._model._actor = model_actor
        self._model._critic = model_critic
        self._modelTarget._actor = modelTarget_actor
        self._modelTarget._critic = modelTarget_critic
        # print ("self._model._actor_train: ", self._model._actor_train)
        
    def loadFrom(self, fileName):
        from keras.models import load_model
        def pos_y(true_y, pred_y):
            return self._actLoss
        suffix = ".h5"
        print ("Loading agent: ", fileName)
        # with K.get_session().graph.as_default() as g:
        self._model._actor = load_model(fileName+"_actor"+suffix)
        self._model._critic = load_model(fileName+"_critic"+suffix)
        self._modelTarget._actor = load_model(fileName+"_actor_T"+suffix)
        self._modelTarget._critic = load_model(fileName+"_critic_T"+suffix)
        # self._model._actor_train = load_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        # self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        hf.close()
        

