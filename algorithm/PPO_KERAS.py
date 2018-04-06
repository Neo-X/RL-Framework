import theano
from theano import tensor as T

import numpy as np
# import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import norm_state, scale_state, norm_action, scale_action, action_bound_std, scale_reward
from algorithm.AlgorithmInterface import AlgorithmInterface
from model.LearningUtil import loglikelihood, likelihood, likelihoodMEAN, kl, kl_D, entropy, flatgrad, zipsame, get_params_flat, setFromFlat
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras

# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class PPO_KERAS(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(PPO_KERAS,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        ## primary network
        self._model = model

        ## Target network
        self._modelTarget = copy.deepcopy(model)
        # self._modelTarget = model
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._Anneal = T.scalar("Anneal")
        
        self._value = self._model.getCriticNetwork()([self._model.getStateSymbolicVariable()])
        self._value_Target = self._modelTarget.getCriticNetwork()([self._model.getResultStateSymbolicVariable()])
        
        _target = self._model.getRewardSymbolicVariable() + (self._discount_factor * self._value_Target)
        self._loss = T.mean(0.5 * (self._value - _target) ** 2)
        
        
        self._q_valsActA = self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,:self._action_length]
        if ( 'use_stocastic_policy' in self.getSettings() and ( self.getSettings()['use_stocastic_policy'])): 
            self._q_valsActASTD = ( T.ones_like(self._q_valsActA)) * self.getSettings()['exploration_rate']
            # self._q_valsActASTD = ( T.ones_like(self._q_valsActA)) * self.getSettings()['exploration_rate']
        else:
            self._q_valsActASTD = (self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) + 1e-2
        
        self._q_valsActTarget_State = self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,:self._action_length]
        if ( 'use_stocastic_policy' in self.getSettings() and ( self.getSettings()['use_stocastic_policy'])): 
            self._q_valsActTargetSTD = (T.ones_like(self._q_valsActTarget_State)) * self.getSettings()['exploration_rate'] 
            # self._q_valsActTargetSTD = (self._action_std_scaling * T.ones_like(self._q_valsActTarget)) * self.getSettings()['exploration_rate']
        else: 
            self._q_valsActTargetSTD = (self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) + 1e-2
        
        self._Advantage = T.col("Advantage")
        
        self._actor_entropy = 0.5 * T.mean((2 * np.pi * self._q_valsActASTD ) )
        
        ## Compute on-policy policy gradient
        self._prob = likelihood(self._model.getActionSymbolicVariable(), self._q_valsActA, self._q_valsActASTD, self._action_length)
        ### How should this work if the target network is very odd, as in not a slightly outdated copy.
        self._prob_target = likelihood(self._model.getActionSymbolicVariable(), self._q_valsActTarget_State, self._q_valsActTargetSTD, self._action_length)
        ## This does the sum already
        self._r = (self._prob / self._prob_target)
        self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((self._r), self._Advantage)
        ppo_epsilon = self.getSettings()['kl_divergence_threshold']
        self._actLoss_2 = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((theano.tensor.clip(self._r, 1.0 - (ppo_epsilon * self._Anneal), 1+ (ppo_epsilon * self._Anneal)), self._Advantage))
        self._actLoss_ = theano.tensor.minimum((self._actLoss_), (self._actLoss_2))
        # self._actLoss = ((T.mean(self._actLoss_) )) + -self._actor_regularization
        # self._actLoss = (-1.0 * (T.mean(self._actLoss_) + (self.getSettings()['std_entropy_weight'] * self._actor_entropy )))
        self._actLoss = -1.0 * T.mean(self._actLoss_)
        if ("ppo_use_seperate_nets" in self.getSettings() and ( self.getSettings()["ppo_use_seperate_nets"] == False)):
            self._actLoss = self._actLoss + self._loss  
        
        # self._policy_grad = T.grad(self._actLoss ,  self._actionParams)
        
        PPO_KERAS.compile(self)
        
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['critic_learning_rate']), beta_1=np.float32(0.9), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print ("Clipping: ", sgd.decay)
        print("sgd, critic: ", sgd)
        self._model.getCriticNetwork().compile(loss='mse', optimizer=sgd)
        # sgd = SGD(lr=0.0005, momentum=0.9)
        
        def neg_y(true_y, pred_y):
            return -pred_y
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['learning_rate']), beta_1=np.float32(0.9), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print("sgd, actor: ", sgd)
        print ("Clipping: ", sgd.decay)
        self._model.getActorNetwork().compile(loss=neg_y, optimizer=sgd)
        
        if ("ppo_use_seperate_nets" in self.getSettings() and ( self.getSettings()["ppo_use_seperate_nets"] == False)):
            self.trainPolicy = theano.function([self._model.getStateSymbolicVariable(),
                                                 self._model.getActionSymbolicVariable(),
                                                 self._model.getResultStateSymbolicVariable(),
                                                 self._model.getRewardSymbolicVariable(),
                                                 self._Advantage,
                                                 self._Anneal  
                                                 # ,K.learning_phase()
                                                 ], [self._actLoss, self._r], 
                            updates= adam_updates(self._actLoss, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate * self._Anneal).items()
                            # ,on_unused_input='warn'
                            # updates= adam_updates(self._actLoss, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate).items()
                            )
        else:
            self.trainPolicy = theano.function([self._model.getStateSymbolicVariable(),
                                                 self._model.getActionSymbolicVariable(),
                                                 self._Advantage,
                                                 self._Anneal,  
                                                 K.learning_phase()
                                                 ], [self._actLoss, self._r], 
                            updates= adam_updates(self._actLoss, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate * self._Anneal).items()
                            ,on_unused_input='warn'
                            # updates= adam_updates(self._actLoss, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate).items()
                            )
        
        self._r = theano.function([self._model.getStateSymbolicVariable(),
                                             self._model.getActionSymbolicVariable(),
                                             # self._Anneal
                                             K.learning_phase()
                                             ], 
                                  [self._r],
                                  on_unused_input='warn')
        
        gradients = K.gradients(T.mean(self._value), [self._model.getStateSymbolicVariable()]) # gradient tensors
        self._get_gradients = K.function(inputs=[self._model.getStateSymbolicVariable(),  K.learning_phase()], outputs=gradients)
        
        self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self._value])
        self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self._value_Target])
        
        self._policy_mean = K.function([self._model.getStateSymbolicVariable(), 
                                          K.learning_phase()], [self._q_valsActA])
        self.q_valsActASTD = K.function([self._model.getStateSymbolicVariable(), 
                                          # self._Anneal,
                                          K.learning_phase()], [self._q_valsActASTD]) 
        

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
        """
        for i in range(len(params[0])):
            params[0][i] = np.array(params[0][i], dtype=self._settings['float_type'])
            """
        self._model.getCriticNetwork().set_weights(params[0])
        self._model.getActorNetwork().set_weights( params[1] )
        self._modelTarget.getCriticNetwork().set_weights( params[2])
        self._modelTarget.getActorNetwork().set_weights( params[3])
    
    def setData(self, states, actions, rewards, result_states, fallen):
        pass
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
    def trainCritic(self, states, actions, rewards, result_states, falls):
        if ("ppo_use_seperate_nets" in self.getSettings() and ( self.getSettings()["ppo_use_seperate_nets"] == False)):
            return 0
        self.setData(states, actions, rewards, result_states, falls)
        # print ("Performing Critic trainning update")
        
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        # y_ = self._modelTarget.getCriticNetwork().predict(result_states, batch_size=states.shape[0])
        y_ = self._value_Target([result_states,0])[0]
        # v = self._model.getCriticNetwork().predict(states, batch_size=states.shape[0])
        # target_ = rewards + ((self._discount_factor * y_) * falls)
        target_ = rewards + ((self._discount_factor * y_))
        target_ = np.array(target_, dtype=self._settings['float_type'])
        # states = np.array(states, dtype=self._settings['float_type'])
        # print ("target type: ", target_.dtype)
        # print ("states type: ", states.dtype)
        # print ("Critic Target: ", np.concatenate((v, target_, rewards, y_) ,axis=1) )
        score = self._model.getCriticNetwork().fit(states, target_,
              nb_epoch=1, batch_size=32,
              verbose=0
              # callbacks=[early_stopping],
              )
        loss = score.history['loss'][0]
        # print(" Critic loss: ", loss)
        
        return loss
    
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions=None, 
                   forwardDynamicsModel=None, p=1.0):
        lossActor = 0
        # print ("PPO p:", p)
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        """
        score = self._model.getActorNetwork().fit([states, actions, advantage], np.zeros_like(rewards),
              nb_epoch=1, batch_size=32,
              verbose=0
              # callbacks=[early_stopping],
              )
        """
        
        if ( (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train'] ) 
             and False 
             ):
            mbae_actions=[]
            mbae_advantage=[]
            other_actions=[]
            other_advantage=[]
            policy_mean = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])[:,:self._action_length]
            # print ("exp_actions: ", exp_actions)
            for k in range(actions.shape[0]):
                if (exp_actions[k] == 2):
                    mbae_actions.append(actions[k]-policy_mean[k])
                    mbae_advantage.append(advantage[k])
                else:
                    other_actions.append(actions[k]-policy_mean[k])
                    other_advantage.append(advantage[k])
            
            
            policy_mean = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])[:,:self._action_length]
            print ("MBAE Actions: ", len(mbae_actions), ", ", len(mbae_actions)/actions.shape[0], "%")
            print ("MBAE Actions std: ", np.std(mbae_actions, axis=0), " mean ", np.mean(np.std(mbae_actions, axis=0)))
            print ("MBAE Actions advantage: ", np.mean(mbae_advantage, axis=0))
            print ("Normal Actions std: ", np.std(other_actions, axis=0), " mean ", np.mean(np.std(other_actions, axis=0)))
            print ("Normal Actions advantage: ", np.mean(other_advantage, axis=0))
        
        # r_ = np.mean(self._r(states, actions, p, 0))
        # r_ = np.mean(self._r(states, actions, p))
        r_ = np.mean(self._r(states, actions, 0))
        
        std = np.std(advantage)
        mean = np.mean(advantage)
        if ( 'advantage_scaling' in self.getSettings() and ( self.getSettings()['advantage_scaling'] != False) ):
            std = std / self.getSettings()['advantage_scaling']
            mean = 0.0
        advantage = (advantage - mean) / std
        ### check to not perform updates when r gets to large.
        et_factor = 1.2
        if (r_ < (et_factor)) and ( r_ > (1.0/et_factor)):  ### update not to large
            # lossActor = score.history['loss'][0]
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print ("Policy probability ratio: ", np.mean(r_))
                print ("Policy mean: ", np.mean(self._policy_mean([states, 0])[0], axis=0))
                print ("Policy std: ", np.mean(self.q_valsActASTD([states, 0])[0], axis=0))
                # print ("States shape: ", np.array(states).shape)
                        ### For now don't include dropout in policy updates
            if ("ppo_use_seperate_nets" in self.getSettings() and ( self.getSettings()["ppo_use_seperate_nets"] == False)):
                (lossActor, r_) = self.trainPolicy(states, actions, result_states, rewards, advantage, p)
            else: 
                (lossActor, r_) = self.trainPolicy(states, actions, advantage, p, 0)
            # (lossActor, r_) = self.trainPolicy(states, actions, advantage, 1.0)
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                r_ = np.mean(self._r(states, actions, 0))
                print ("Policy loss: ", lossActor, " r: ", np.mean(r_))
            
            if ( (not np.isfinite(lossActor)) or (not np.isfinite(np.mean(r_)))):
                print("Something bad happend go back to the old policy")
                print ("State mean: ", np.mean(states, axis=0))
                print ("Actions mean: ", np.mean(actions, axis=0))
                print ("Advantage mean: ", np.mean(advantage, axis=0))
                self._model.getCriticNetwork().set_weights( copy.deepcopy(self._modelTarget.getCriticNetwork().get_weights()))
                self._model.getActorNetwork().set_weights( copy.deepcopy(self._modelTarget.getActorNetwork().get_weights()))
                if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                    r_ = np.mean(self._r(states, actions, 0))
                    print ("Policy probability ratio: ", np.mean(r_))
                    print ("Policy mean: ", np.mean(self._policy_mean([states, 0])[0], axis=0))
                    print ("Policy std: ", np.mean(self.q_valsActASTD([states, 0])[0], axis=0))
        else:
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print ("Policy Gradient too large: ", np.mean(r_))
            
        return lossActor
    
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
        self._model.setStates(state)
        # action_ = lasagne.layers.get_output(self._model.getActorNetwork(), state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        action_ = scale_action(self._model.getActorNetwork().predict(state, batch_size=1)[:,:self._action_length], self._action_bounds)
        # action_ = scale_action(self._q_action_target()[0], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def predict_std(self, state, deterministic_=True, p=1.0):
        # print ("PPO std p:", p)
        state = norm_state(state, self._state_bounds)   
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            action_std = self.q_valsActASTD([state, 0])[0]
            # action_std = self._q_action_std()[0] * (action_bound_std(self._action_bounds))
        else:
            action_std = self.q_valsActASTD([state, 0])[0] * (action_bound_std(self._action_bounds))
        return action_std
    
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

    """
    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        # return scale_reward(self._q_valTarget(), self.getRewardBounds())[0]
        value = scale_reward(self._model.getCriticNetwork().predict(state, batch_size=1), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        print ("value: ", repr(np.array(value)))
        return value
        # return self._q_val()[0]
    
    def q_values(self, state):
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        values = self._model.getCriticNetwork().predict(state, batch_size=state.shape[0])
        print ("values: ", repr(np.array(values)))
        return values    
    """
    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        # return scale_reward(self._q_valTarget(), self.getRewardBounds())[0]
        value = scale_reward(self._value([state,0])[0], self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # print ("value: ", repr(np.array(value)))
        return value
        # return self._q_val()[0]
    
    def q_values(self, states):
        states = np.array(states, dtype=self._settings['float_type'])
        values = self._value([states,0])[0]
        # print ("values: ", repr(np.array(values)))
        return values
    
    def q_valueWithDropout(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = np.array(state, dtype=self._settings['float_type'])
        state = norm_state(state, self._state_bounds)
        self._model.setStates(state)
        return scale_reward(self._q_val_drop(), self.getRewardBounds())
    
    def bellman_error(self, states, actions, rewards, result_states, falls):
        """
            Computes the one step temporal difference.
        """
        y_ = self._value_Target([result_states,0])[0]
        # y_ = self._modelTarget2.getValueFunction().predict(result_states, batch_size=states.shape[0])
        target_ = rewards + ((self._discount_factor * y_))
        # values =  self._model.getValueFunction().predict(states, batch_size=states.shape[0])
        values = self._value([states,0])[0]
        bellman_error = target_ - values
        return bellman_error
        # return self._bellman_errorTarget()
        
from collections import OrderedDict
def adam_updates(loss, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):

    all_grads = T.grad(loss, params)
    t_prev = theano.shared(np.array(0,dtype="float64"))
    updates = OrderedDict()

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(1-beta2**t)/(1-beta1**t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (1-beta1)*g_t
        v_t = beta2*v_prev + (1-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates