
import numpy as np
# import lasagne
import sys
import copy
sys.path.append('../')
from algorithm.AlgorithmInterface import AlgorithmInterface

# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class ModelDummy(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(ModelDummy,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        
        
        ModelDummy.compile(self)
        
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        pass 
    
    def sample(self, state_, deterministic_=True, evaluation_=False, p=None, sim_index=None, bootstrapping=False,
               epsilon=1.0, sampling=False, time_step=0):
        action = self.predict(state_)
        return (action, [1], [0], state_)
    
    def getGrads(self, states, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        grads = 0
        return grads
        
    def updateTargetModel(self):
        print ("Updating target Model")
        """
            Target model updates
        """
        #  self._modelTarget.getCriticNetwork().set_weights( copy.deepcopy(self._model.getCriticNetwork().get_weights()))
        # self._modelTarget.getActorNetwork().set_weights( copy.deepcopy(self._model.getActorNetwork().get_weights()))
        pass
        
    def getNetworkParameters(self):
        params = []
        return params
    
    def setNetworkParameters(self, params):
        """
        for i in range(len(params[0])):
            params[0][i] = np.array(params[0][i], dtype=self._settings['float_type'])
            """
        i=0
    
    def setData(self, states, actions, rewards, result_states, fallen):
        pass
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
    def trainCritic(self, states, actions, rewards, result_states, falls):
        loss = 0
        return loss
    
    def trainActor(self, states, actions, rewards, result_states, falls, advantage,
                    exp_actions=None, forwardDynamicsModel=None, p=1.0):
        lossActor = 0
        
        return lossActor
    
    def train(self, states, actions, rewards, result_states, falls):
        loss = 0
        return loss
    
    def predict(self, state, deterministic_=True, evaluation_=False, p=None, sim_index=None, bootstrapping=False):
        
        action_ = [[0] * self._action_length]
        return action_
    
    def predict_std(self, state, deterministic_=True, p=1.0):
        action_std = [[0.1] * self._action_length]
        return action_std
    
    def predictWithDropout(self, state, deterministic_=True):
        action_ = [[0] * self._action_length]
        return action_
    
    def q_value(self, state):
        value = [np.array([0])]
        return value
        # return self._q_val()[0]
    
    def q_values(self, state):
        """
            For returning a vector of q values, state should already be normalized
        """
        values_ = np.ones((states.shape[0], 1))
        return values_
    
    def q_values2(self, states, wrap=True):
        """
            For returning a vector of q values, state should already be normalized
        """
        # print ("len(states) shape: ", np.array(states).shape)
        values_ = np.ones((states.shape[0], 1))
        # print ("values_ shape: ", np.array(values_).shape)
        return values_
    
    def q_valueWithDropout(self, state):
        values_ = [[0.8]]
        return values_
    
    def bellman_error(self, states, actions, rewards, result_states, falls):
        values_ = [[0.3]]
        return values_
        # return self._bellman_errorTarget()
        
    def trainDyna(self, predicted_states, actions, rewards, result_states, falls):
        loss = [0]
        # print(" Critic loss: ", loss)
        
    def get_actor_regularization(self):
        return 0
    
    def get_actor_loss(self):
        return 0
    
    def get_critic_regularization(self):
        return 0
    
    def get_critic_loss(self):
        return 0
    
    def setNoise(self, noise):
        pass
