
import numpy as np
# import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import *
from algorithm.KERASAlgorithm import KERASAlgorithm
from model.LearningUtil import loglikelihood, kl, entropy, change_penalty
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras
from keras.models import Sequential, Model

from model.LearningUtil import loglikelihood_keras, likelihood_keras, kl_keras, kl_D_keras, entropy_keras

# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class PolicyDistribution(KERASAlgorithm):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False):

        super(PolicyDistribution,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False)
        
        ## primary network
        self._model = model
        self._model._actor = Model(inputs=self._model.getStateSymbolicVariable(), outputs=self._model._actor)
        print("Actor summary: ", self._model._actor.summary())
        self._model._critic = Model(inputs=self._model.getStateSymbolicVariable(), outputs=self._model._critic)
        print("Critic summary: ", self._model._critic.summary())
        
        ### Target network
        # self._modelTarget = copy.deepcopy(model)
        self._modelTarget = type(self._model)(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False)
        self._modelTarget._actor = Model(inputs=self._modelTarget.getStateSymbolicVariable(), outputs=self._modelTarget._actor)
        print("Target Actor summary: ", self._modelTarget._actor.summary())
        self._modelTarget._critic = Model(inputs=self._modelTarget.getStateSymbolicVariable(), outputs=self._modelTarget._critic)
        print("Target Critic summary: ", self._modelTarget._critic.summary())
        # print ("Loss ", self._model.getActorNetwork().total_loss)
        
        self._q_valsActA = self._model.getActorNetwork()(self._model._stateInput)
        self._q_valsActTarget = self._modelTarget.getActorNetwork()(self._model._stateInput)
        
        self._q_valsActA = self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,:self._action_length]
        if ( 'use_stochastic_policy' in self.getSettings() and ( self.getSettings()['use_stochastic_policy'])): 
            # self._q_valsActASTD = (self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) + 1e-2
            self._q_valsActASTD = ((self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) * self.getSettings()['exploration_rate']) + 1e-2
        else:
            self._q_valsActASTD = ( K.ones_like(self._q_valsActA)) * self.getSettings()['exploration_rate']
        
        self._q_valsActTarget_State = self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,:self._action_length]
        if ( 'use_stochastic_policy' in self.getSettings() and ( self.getSettings()['use_stochastic_policy'])): 
            # self._q_valsActTargetSTD = (self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) + 1e-2
            self._q_valsActTargetSTD = ((self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) * self.getSettings()['exploration_rate']) + 1e-2 
        else:
            self._q_valsActTargetSTD = (K.ones_like(self._q_valsActTarget_State)) * self.getSettings()['exploration_rate']
        
        self.__value = self._model.getCriticNetwork()([self._model.getStateSymbolicVariable()])
        self.__value_Target = self._modelTarget.getCriticNetwork()([self._model.getResultStateSymbolicVariable()])
        
        _target = self._model.getRewardSymbolicVariable() + (self._discount_factor * self.__value_Target)
        self._loss = K.mean(0.5 * (self.__value - _target) ** 2)
        
        PolicyDistribution.compile(self)
        
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        
        def loss_std(action_true, action_pred):
            action_true = action_true[:,:self._action_length]
            action_pred_mean = action_pred[:,:self._action_length]
            action_pred_std = action_pred[:,self._action_length:]
            prob = loglikelihood_keras(action_true, action_pred_mean, action_pred_std, self._action_length)
            # entropy = 0.5 * T.mean(T.log(2 * np.pi * action_pred_std + 1 ) )
            # actLoss = -1.0 * (K.mean(K.mean(prob, axis=-1)) + (entropy * 1e-2))
            actLoss = -1.0 * (K.mean(K.mean(prob, axis=-1)))
            ### Average over batch
            return actLoss
        
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['critic_learning_rate']), beta_1=np.float32(0.9), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print ("Clipping: ", sgd.decay)
        print("sgd, critic: ", sgd)
        self._model.getCriticNetwork().compile(loss='mse', optimizer=sgd)
        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['learning_rate']), beta_1=np.float32(0.9), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print("sgd, actor: ", sgd)
        print ("Clipping: ", sgd.decay)
        if ("use_stochastic_policy" in self.getSettings()
            and (self.getSettings()['use_stochastic_policy'] == True)):
            self._model.getActorNetwork().compile(loss=loss_std, optimizer=sgd)
        else:
            self._model.getActorNetwork().compile(loss='mse', optimizer=sgd)
            
        self._q_action_std = K.function([self._model._stateInput], [self._q_valsActASTD])
        
        gradients = K.gradients(K.mean(self.__value), [self._model._stateInput]) # gradient tensors
        self._get_gradients = K.function(inputs=[self._model._stateInput,  K.learning_phase()], outputs=gradients)
        
        if (self.getSettings()["regularization_weight"] > 0.0000001):
            self._actor_regularization = K.sum(self._model.getActorNetwork().losses)
        else:
            self._actor_regularization = K.sum(self._model.getActorNetwork().losses)
        
        if (self.getSettings()["critic_regularization_weight"] > 0.0000001):
            self._critic_regularization = K.sum(self._model.getCriticNetwork().losses)
        else:
            self._critic_regularization = K.sum(self._model.getCriticNetwork().losses)
            
        print ("build regularizers")
        self._get_actor_regularization = K.function([], [self._actor_regularization])
        self._get_critic_regularization = K.function([], [self._critic_regularization])
        self._get_critic_loss = K.function([self._model.getStateSymbolicVariable(),
                                            self._model.getRewardSymbolicVariable(), 
                                            self._model.getResultStateSymbolicVariable(),
                                            K.learning_phase()], [self._loss])
        
        self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        
    def trainActor(self, states, actions, rewards, result_states, falls, advantage,
                    exp_actions=None, G_t=[[0]], forwardDynamicsModel=None, p=1.0, updates=1, batch_size=None):
        if (batch_size is None):
            batch_size_=states.shape[0]
        else:
            batch_size_=batch_size
        lossActor = 0
        
        
        score = self._model.getActorNetwork().fit(np.array(states), np.array(actions),
          epochs=1, batch_size=len(actions),
          verbose=0,
          shuffle=True
          # callbacks=[early_stopping],
              )
        
        lossActor = score.history['loss'][0]
        # print("Policy Loss: ", lossActor)
        return lossActor
    
    
    def predict_std(self, state, deterministic_=True, p=1.0):
        state = norm_state(state, self._state_bounds)   
        state = np.array(state, dtype=self._settings['float_type'])
        
        # action_std = self._model.getActorNetwork().predict(state, batch_size=1)[:,self._action_length:] * (action_bound_std(self._action_bounds))
        action_std = self._q_action_std([state])[0] * action_bound_std(self._action_bounds)
        # print ("Policy std: ", action_std)
        return action_std
    
    def bellman_error(self, states, actions, rewards, result_states, falls):
        """
            Computes the one step temporal difference.
        """
        y_ = self._modelTarget.getCriticNetwork().predict(result_states, batch_size=states.shape[0])
        target_ = rewards + ((self._discount_factor * y_))
        if self._settings['on_policy']:
            values =  self._modelTarget.getCriticNetwork().predict(states, batch_size=states.shape[0])
        else:
            values =  self._model.getCriticNetwork().predict(states, batch_size=states.shape[0])
        bellman_error = target_ - values
        return bellman_error
        # return self._bellman_errorTarget()
        
    def get_actor_regularization(self):
        return self._get_actor_regularization([])
    
    def get_actor_loss(self, state, action, reward, nextState, advantage):
        return 0
    
    def get_critic_regularization(self):
        return self._get_critic_regularization([])
    
    def get_critic_loss(self, state, action, reward, nextState):
        return self._get_critic_loss([state, reward, nextState, 0])
