import numpy as np
import h5py
# import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import norm_state, scale_state, norm_action, scale_action, action_bound_std, scale_reward, norm_reward
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

    def getGrads(self, states, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if ( alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=self._settings['float_type'])
        # grads = np.reshape(np.array(self._get_gradients([states])[0], dtype=self._settings['float_type']), (states.shape[0],states.shape[1]))
        grads = np.array(self._get_gradients([states, 0]), dtype=self._settings['float_type'])
        # print ("State grads: ", grads.shape)
        # print ("State grads: ", repr(grads))
        return grads
    
    def setData(self, states, actions, rewards, result_states, fallen):
        pass
        
    def updateTargetModel(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Updating target Model")
        """
            Target model updates
        """
        self._modelTarget.getCriticNetwork().set_weights( copy.deepcopy(self._model.getCriticNetwork().get_weights()))
        self._modelTarget.getActorNetwork().set_weights( copy.deepcopy(self._model.getActorNetwork().get_weights()))
        
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._model.getActorNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget.getActorNetwork().get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        self._model.getCriticNetwork().set_weights(params[0])
        self._model.getActorNetwork().set_weights( params[1] )
        self._modelTarget.getCriticNetwork().set_weights( params[2])
        self._modelTarget.getActorNetwork().set_weights( params[3])
        
    def trainCritic(self, states, actions, rewards, result_states, falls, G_t=[[0]],
                    updates=1, batch_size=None):
        
        if ("use_single_network" in self.getSettings() and ( self.getSettings()["use_single_network"] == True)):
            # print("self.getSettings()[\"use_single_network\"]: ", self.getSettings()["use_single_network"])
            return 0

        if (batch_size is None):
            batch_size_=states.shape[0]
        else:
            batch_size_=batch_size
        
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        if ('dont_use_td_learning' in self.getSettings() 
            and self.getSettings()['dont_use_td_learning'] == True):
            y_ = self._value_Target([result_states,0])[0]
            target_ = rewards + ((self._discount_factor * y_))
            target_2 = norm_reward(G_t, self.getRewardBounds()) * (1.0-self.getSettings()['discount_factor'])
            target = (target_ + target_2) / 2.0
        else:
            # y_ = self._modelTarget.getCriticNetwork().predict(result_states, batch_size=states.shape[0])
            y_ = self._value_Target([result_states,0])[0]
            # v = self._model.getCriticNetwork().predict(states, batch_size=states.shape[0])
            # target_ = rewards + ((self._discount_factor * y_) * falls)
            target_ = rewards + ((self._discount_factor * y_))
        # y_ = self._modelTarget.getCriticNetwork().predict(result_states, batch_size=states.shape[0])
        # target_ = rewards + ((self._discount_factor * y_) * falls)
        target_ = np.array(target_, dtype=self._settings['float_type'])
        # states = np.array(states, dtype=self._settings['float_type'])
        # print ("target type: ", target_.dtype)
        # print ("states type: ", states.dtype)
        v = self._model.getCriticNetwork().predict(states, batch_size=states.shape[0])
        # print ("Critic Target: ", np.concatenate((v, target_, rewards, y_) ,axis=1) )
        c_error = np.mean(np.mean(np.square(v - target_), axis=1))
        # print ("critic error: ", np.mean(np.mean(np.square(v - target_), axis=1)))
        # if (c_error < 10.0):
        score = self._model.getCriticNetwork().fit(states, target_,
              epochs=updates, batch_size=batch_size_,
              verbose=0,
              shuffle=True
              # callbacks=[early_stopping],
              )
        loss = score.history['loss'][0]
        #else:
        #    print ("Critic error to high:", c_error)
        #    loss = 0
        # print(" Critic loss: ", loss)
        
        return loss
    
    def train(self, states, actions, rewards, result_states, falls):
        loss = self.trainCritic(states, actions, rewards, result_states, falls)
        lossActor = self.trainActor(states, actions, rewards, result_states, falls)
        return loss
    
    def predict(self, state, deterministic_=True, evaluation_=False, p=None, sim_index=None, bootstrapping=False):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        # state = np.array(state, dtype=self._settings['float_type'])
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        # self._model.setStates(state)
        # print("state: ", repr(state))
        # action_ = lasagne.layers.get_output(self._model.getActorNetwork(), state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        action_ = scale_action(self._model.getActorNetwork().predict(state, batch_size=1)[:,:self._action_length], self._action_bounds)
        # action_ = scale_action(self._model.getActorNetwork().predict([state, np.zeros((1,2)), np.zeros((1,1)), np.zeros((1,1))], batch_size=1)[:,:self._action_length], self._action_bounds)
        # action_ = scale_action(self._q_action_target()[0], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def predictWithDropout(self, state, deterministic_=True):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = np.array(state, dtype=self._settings['float_type'])
        state = norm_state(state, self._state_bounds)
        self._model.setStates(state)
        # action_ = lasagne.layers.get_output(self._model.getActorNetwork(), state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        action_ = scale_action(self._model.getActorNetwork().predict(states, batch_size=1)[:,:self._action_length], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        # return scale_reward(self._q_valTarget(), self.getRewardBounds())[0]
        value = scale_reward(self._model.getCriticNetwork().predict(state), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # value = scale_reward(self._value([state,0])[0], self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # print ("value: ", repr(np.array(value)))
        return value
        # return self._q_val()[0]
            
    def q_values(self, states):
        states = np.array(states, dtype=self._settings['float_type'])
        # print("states: ", repr(states))
        values = self._model.getCriticNetwork().predict(states)
        # values = self._value([states,0])[0]
        # print ("values: ", repr(np.array(values)))
        return values
    
    def q_valueWithDropout(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = np.array(state, dtype=self._settings['float_type'])
        state = norm_state(state, self._state_bounds)
        self._model.setStates(state)
        return scale_reward(self._q_val_drop(), self.getRewardBounds())
        
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
        if (self._modelTarget is not None):
            self._modelTarget._actor.save(fileName+"_actor_T"+suffix, overwrite=True)
            self._modelTarget._critic.save(fileName+"_critic_T"+suffix, overwrite=True)
        # print ("self._model._actor_train: ", self._model._actor_train)
        
    def loadFrom(self, fileName):
        from keras.models import load_model
        suffix = ".h5"
        print ("Loading agent: ", fileName)
        # with K.get_session().graph.as_default() as g:
        self._model._actor = load_model(fileName+"_actor"+suffix)
        self._model._critic = load_model(fileName+"_critic"+suffix)
        if (self._modelTarget is not None):
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
        

