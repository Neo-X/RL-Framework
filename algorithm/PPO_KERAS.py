import numpy as np
# import lasagne
import h5py
import sys
import copy
sys.path.append('../')
from model.ModelUtil import norm_state, scale_state, norm_action, scale_action, action_bound_std, scale_reward, norm_reward
from algorithm.KERASAlgorithm import *
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
    
    

class PPO_KERAS(KERASAlgorithm):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False):

        super(PPO_KERAS,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=print_info)
        
        print ("PPO_KERAS: created models")
        self._modelTarget = type(self._model)(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=print_info)
        
        # self._modelTarget = model
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._Anneal = keras.layers.Input(shape=(1,), name="Anneal")
        self._Advantage = keras.layers.Input(shape=(1,), name="Advantage")
        
        self._PoliAction = keras.layers.Input(shape=(self._action_length,), name="PoliAction")
        if ( 'use_stochastic_policy' in self.getSettings() and ( self.getSettings()['use_stochastic_policy'])):
            self._PoliAction = keras.layers.Input(shape=(self._action_length*2,), name="PoliAction")
        
        ## primary network
        self._model = model
        input_ = [self._model.getStateSymbolicVariable(),
                 self._PoliAction,
                 self._Advantage,
                 self._Anneal
                  ]
        if ("policy_latent_length" in self.getSettings()):
            self._PoliAction2 = keras.layers.Input(shape=(self._action_length,), name="PoliAction2")
            input_ = [self._model.getStateSymbolicVariable(),
                 self._PoliAction,
                 self._Advantage,
                 self._Anneal,
                 self._model.getResultStateSymbolicVariable(),
                 self._PoliAction2
                  ]
        
        if ("use_single_network" in self.getSettings() and ( self.getSettings()["use_single_network"] == True)):
            self._model._actor_train = Model(inputs=input_, outputs=[self._model._actor, self._model._critic])
        else:
            self._model._actor_train = Model(inputs=input_, outputs=[self._model._actor])
        self._model._actor = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=[self._model._actor])
        if (print_info):
            print("Actor summary: ", self._model._actor_train.summary())
            
        if ("force_use_result_state_for_critic" in self._settings
            and (self._settings["force_use_result_state_for_critic"] == True)):
            inputs_ = [self._model.getResultStateSymbolicVariable()]
            inputs_target = [self._modelTarget.getResultStateSymbolicVariable()]
        else:
            inputs_ = [self._model.getStateSymbolicVariable()]
            inputs_target = [self._modelTarget.getStateSymbolicVariable()]
            
        print ("critic input: ", repr(inputs_))
        print ("critic inputs_target: ", repr(inputs_target))
        self._model._critic = Model(inputs=inputs_, outputs=[self._model._critic])
        if (print_info):
            print("Critic summary: ", self._model._critic.summary())
        ## Target network
        input_Target = [self._modelTarget.getStateSymbolicVariable(),
                 self._PoliAction,
                 self._Advantage,
                 self._Anneal
                  ]
        self._modelTarget._actor = Model(inputs=[self._modelTarget.getStateSymbolicVariable()], outputs=[self._modelTarget._actor])
        if (print_info):
            print("Target Actor summary: ", self._modelTarget._actor.summary())
        self._modelTarget._critic = Model(inputs=inputs_target, outputs=[self._modelTarget._critic])
        if (print_info):
            print("Target Critic summary: ", self._modelTarget._critic.summary())
        
        
        PPO_KERAS.compile(self)
        self.updateTargetModel()
        
    def compile(self):
        if ("force_use_result_state_for_critic" in self._settings
            and (self._settings["force_use_result_state_for_critic"] == True)):
            inputs_ = [self._model.getResultStateSymbolicVariable()]
            inputs_target = [self._modelTarget.getResultStateSymbolicVariable()]
        else:
            inputs_ = [self._model.getStateSymbolicVariable()]
            inputs_target = [self._modelTarget.getStateSymbolicVariable()]
        print ("self._model.getStateSymbolicVariable(): ", repr(self._model.getStateSymbolicVariable()))
        self.__value = self._model.getCriticNetwork()(inputs_)
        self.__value_Target = self._modelTarget.getCriticNetwork()(inputs_target)
        
        _target = self._model.getRewardSymbolicVariable() + (self._discount_factor * self.__value_Target)
        self._loss = K.mean(0.5 * (self.__value - _target) ** 2)
        
        
        self._q_valsActA = self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,:self._action_length]
        if ( 'use_stochastic_policy' in self.getSettings() and ( self.getSettings()['use_stochastic_policy'])): 
            self._q_valsActASTD = ((self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) * self.getSettings()['exploration_rate']) + 1e-2
        else:
            self._q_valsActASTD = ( K.ones_like(self._q_valsActA)) * self.getSettings()['exploration_rate']
        
        self._q_valsActTarget_State = self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,:self._action_length]
        if ( 'use_stochastic_policy' in self.getSettings() and ( self.getSettings()['use_stochastic_policy'])): 
            self._q_valsActTargetSTD = ((self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) * self.getSettings()['exploration_rate']) + 1e-2 
        else:
            self._q_valsActTargetSTD = (K.ones_like(self._q_valsActTarget_State)) * self.getSettings()['exploration_rate']
        
        self._actor_entropy = entropy_keras(self._q_valsActASTD)
        
        ## Compute on-policy policy gradient
        self._prob = likelihood_keras(self._model.getActionSymbolicVariable(), self._q_valsActA, self._q_valsActASTD, self._action_length)
        ### How should this work if the target network is very odd, as in not a slightly outdated copy.
        self._prob_target = likelihood_keras(self._model.getActionSymbolicVariable(), self._q_valsActTarget_State, self._q_valsActTargetSTD, self._action_length)
        ## This does the sum already
        self.__r = (self._prob / self._prob_target)
        self._actLoss_ = (self.__r) * self._Advantage
        if ("policy_latent_length" in self.getSettings()):
            ### goal reward
            self._pi_z = self._q_valsActA[:,-self.getSettings()["policy_latent_length"]:]
            diff_z = self._model.getResultStateSymbolicVariable() - self._pi_z
            diff_z = K.sum(K.sqrt(diff_z*diff_z),axis=0)
            latent_reward = K.exp(diff_z*diff_z * -5.0)
            ### latent change penalty
            ### This is kind of tricky because z will be from a distribution so this should
            ### cause that distribution to colapse
            self._pi_ss = self._modelTarget.getActorNetwork()(self._model.getResultStateSymbolicVariable())[:,:self._action_length]
            self._pi_zz = self._pi_ss[:,-self.getSettings()["policy_latent_length"]:]
            diff_zz = self._pi_z - self._pi_zz
            ### Mean across row
            l_cost = K.mean(diff_zz*diff_zz, axis=0)
            self._actLoss_ = self._actLoss_ + latent_reward + l_cost
        ppo_epsilon = self.getSettings()['kl_divergence_threshold']
        self._actLoss_2 = (K.clip(self.__r, 1.0 - (ppo_epsilon * self._Anneal), 1 + (ppo_epsilon * self._Anneal)), self._Advantage)
        self._actLoss_ = K.minimum(self._actLoss_, self._actLoss_2)
        self._actLoss = -1.0 * K.mean(self._actLoss_)
        self._actLoss_tmp = self._actLoss
        if ("use_single_network" in self.getSettings() and ( self.getSettings()["use_single_network"] == True)):
            self._actLoss = self._actLoss + self._loss  
        
        # self._policy_grad = T.grad(self._actLoss ,  self._actionParams)
        
        sgd = getOptimizer(lr=np.float32(self.getSettings()['critic_learning_rate']), 
                                    settings=self.getSettings())
        print ("Clipping: ", sgd.decay)
        print("sgd, critic: ", sgd)
        self._model.getCriticNetwork().compile(loss='mse', optimizer=sgd)
        # sgd = SGD(lr=0.0005, momentum=0.9)
        
        def neg_y(true_y, pred_y):
            return -pred_y
        
        def pos_y(true_y, pred_y):
            return self._actLoss
        
        def step_decay(p_):
           initial_lrate = np.float32(self.getSettings()['learning_rate'])
           lrate = initial_lrate * p_
           print ("lrate: ", lrate)
           return lrate
        
        lrate = keras.callbacks.LearningRateScheduler(step_decay)
        self._callbacks_list = [lrate]
        
        def poli_loss_soft(action_old, advantage, anneal, next_state, action_old_next):
            ## Compute on-policy policy gradient
            action_old_mean = action_old[:,:self._action_length]
            if ( 'use_stochastic_policy' in self.getSettings() and ( self.getSettings()['use_stochastic_policy'])):
                action_old_std = action_old[:,self._action_length:]
            else:
                action_old_std = (K.ones_like(action_old_mean)) * self.getSettings()['exploration_rate']
            
            def losssoft2(action_true, action_pred):
                action_true = action_true[:,:self._action_length]
                action_pred_mean = action_pred[:,:self._action_length]
                if ( 'use_stochastic_policy' in self.getSettings() and ( self.getSettings()['use_stochastic_policy'])):
                    action_pred_std = action_pred[:,self._action_length:]
                else:
                    action_pred_std = (K.ones_like(action_pred_mean)) * self.getSettings()['exploration_rate']
                prob = likelihood_keras(action_true, action_pred_mean, action_pred_std, self._action_length)
                prob_target = likelihood_keras(action_true, action_old_mean, action_old_std, self._action_length)
                _r = (prob / prob_target)
                actLoss_ = (_r) * advantage
                ### goal reward
                self._pi_z = action_pred[:,-self.getSettings()["policy_latent_length"]:]
                diff_z = next_state - self._pi_z
                diff_z = K.sum(K.sqrt(diff_z*diff_z),axis=0)
                latent_reward = K.exp(diff_z*diff_z * -5.0)
                ### latent change penalty
                ### This is kind of tricky because z will be from a distribution so this should
                ### cause that distribution to colapse
                self._pi_ss = action_old_next[:,:self._action_length]
                self._pi_zz = self._pi_ss[:,-self.getSettings()["policy_latent_length"]:]
                diff_zz = self._pi_z - self._pi_zz
                ### Mean across row
                l_cost = K.mean(diff_zz*diff_zz, axis=0)
                ppo_epsilon = self.getSettings()['kl_divergence_threshold']
                actLoss_2 = (K.clip(_r, 1.0 - (ppo_epsilon * anneal), 1 + (ppo_epsilon * anneal)), advantage)
                actLoss_ = K.minimum(actLoss_, actLoss_2)
                ### Average across action dimensions
                actLoss = -1.0 * K.mean(actLoss_, axis=-1)
                ### Average over batch
                return K.mean(actLoss)

            self.__loss = losssoft2
            return losssoft2
        def poli_loss(action_old, advantage, anneal):
            ## Compute on-policy policy gradient
            action_old_mean = action_old[:,:self._action_length]
            if ( 'use_stochastic_policy' in self.getSettings() and ( self.getSettings()['use_stochastic_policy'])):
                action_old_std = action_old[:,self._action_length:]
            else:
                action_old_std = (K.ones_like(action_old_mean)) * self.getSettings()['exploration_rate']
            
            def loss2(action_true, action_pred):
                action_true = action_true[:,:self._action_length]
                action_pred_mean = action_pred[:,:self._action_length]
                if ( 'use_stochastic_policy' in self.getSettings() and ( self.getSettings()['use_stochastic_policy'])):
                    action_pred_std = action_pred[:,self._action_length:]
                else:
                    action_pred_std = (K.ones_like(action_pred_mean)) * self.getSettings()['exploration_rate']
                prob = likelihood_keras(action_true, action_pred_mean, action_pred_std, self._action_length)
                prob_target = likelihood_keras(action_true, action_old_mean, action_old_std, self._action_length)
                _r = (prob / prob_target)
                actLoss_ = (_r) * advantage
                ppo_epsilon = self.getSettings()['kl_divergence_threshold']
                actLoss_2 = K.clip(_r, 1.0 - (ppo_epsilon * anneal), 1 + (ppo_epsilon * anneal))* advantage
                actLoss_ = K.minimum(actLoss_, actLoss_2)
                ### Average across action dimensions
                actLoss = -1.0 * K.mean(actLoss_, axis=-1)
                ### Average over batch
                return K.mean(actLoss)

            self.__loss = loss2
            return loss2
        
        self._poli_loss = poli_loss
        sgd = sgd = getOptimizer(lr=np.float32(self.getSettings()['learning_rate']), 
                                    settings=self.getSettings())
        print("sgd, actor: ", sgd)
        print ("Clipping: ", sgd.decay)
        """
        self._m7odel.getActorNetwork().compile(loss=poli_loss(
                    state=self._model.getStateSymbolicVariable(),
                    action=self._model.getActionSymbolicVariable(),
                    advantage=self._Advantage,
                    anneal=self._Anneal), optimizer=sgd)
        """
        if ("use_single_network" in self.getSettings() and ( self.getSettings()["use_single_network"] == True)):
            self._model._actor_train.compile(
                        loss=[poli_loss(action_old=self._PoliAction,
                                        advantage=self._Advantage, 
                                        anneal=self._Anneal), 
                              'mse'], 
                        loss_weights=[0.5, 0.5],
                        optimizer=sgd)
        else:
            if ("policy_latent_length" in self.getSettings()):
                self._model._actor_train.compile(
                        loss=[poli_loss_soft(action_old=self._PoliAction,
                                        advantage=self._Advantage, 
                                        anneal=self._Anneal,
                                        next_state=self._model.getResultStateSymbolicVariable(), 
                                        action_old_next=self._PoliAction2
                                        )], 
                        optimizer=sgd)
            else:
                self._model._actor_train.compile(
                        loss=[poli_loss(action_old=self._PoliAction,
                                        advantage=self._Advantage, 
                                        anneal=self._Anneal)], 
                        optimizer=sgd)
        
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
        if ("policy_latent_length" in self.getSettings()):
            self._get_actor_loss = K.function([self._model.getStateSymbolicVariable(),
                                                 self._model.getActionSymbolicVariable(),
                                                 self._Advantage,
                                                 self._Anneal,
                                                 self._model.getResultStateSymbolicVariable()
                                                 # ,K.learning_phase()
                                                 ], [self._actLoss_tmp])
        else:
            self._get_actor_loss = K.function([self._model.getStateSymbolicVariable(),
                                                 self._model.getActionSymbolicVariable(),
                                                 self._Advantage,
                                                 self._Anneal,
                                                 # self._model.getResultStateSymbolicVariable()
                                                 # ,K.learning_phase()
                                                 ], [self._actLoss_tmp])
        print ("build actor updates")
        
        self._r = K.function([self._model.getStateSymbolicVariable(),
                                     self._model.getActionSymbolicVariable(),
                                     # ,self._Anneal
                                     K.learning_phase()
                                     ], 
                                  [self.__r]
                                  # ,on_unused_input='warn'
                                  )
        
        gradients = K.gradients(K.mean(self.__value), [self._model.getStateSymbolicVariable()]) # gradient tensors
        self._get_gradients = K.function(inputs=[self._model.getStateSymbolicVariable(),  K.learning_phase()], outputs=gradients)
        
        if ("force_use_result_state_for_critic" in self._settings
            and (self._settings["force_use_result_state_for_critic"] == True)):
            inputs_ = [self._model.getResultStateSymbolicVariable(), K.learning_phase()]
            inputs_target = [self._modelTarget.getResultStateSymbolicVariable(), K.learning_phase()]
        else:
            inputs_ = [self._model.getStateSymbolicVariable(), K.learning_phase()]
            inputs_target = [self._modelTarget.getStateSymbolicVariable(), K.learning_phase()]
            
        self._value = K.function(inputs_, [self.__value])
        if ("use_target_net_for_critic" in self.getSettings() and
            (self.getSettings()["use_target_net_for_critic"] == False)):
            self._value_Target = self._value
        else:
            self._value_Target = K.function(inputs_target, [self.__value_Target])
            
        self._action_Target = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self._q_valsActTarget_State])        
        self._policy_mean = K.function([self._model.getStateSymbolicVariable(), 
                                          K.learning_phase()], [self._q_valsActA])
        self._q_action_std = K.function([self._model.getStateSymbolicVariable(), 
                                          # self._Anneal,
                                          K.learning_phase()], [self._q_valsActASTD]) 
        
        print ("Done building PPO KERAS")
        
        
    def trainCritic(self, states, actions, rewards, result_states, falls, G_t=[[0]], p=1.0,
                    updates=1, batch_size=None):
        
        if ("use_single_network" in self.getSettings() and ( self.getSettings()["use_single_network"] == True)):
            # print("self.getSettings()[\"use_single_network\"]: ", self.getSettings()["use_single_network"])
            return 0
        
        loss = super(PPO_KERAS,self).trainCritic(states, actions, rewards, 
                                          result_states, falls, G_t, p,
                                          updates, batch_size)
        return loss
        
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions=None, 
                   G_t=[[0]], forwardDynamicsModel=None, p=1.0, updates=1, batch_size=None):
        lossActor = 0
        self.reset()
        if (batch_size is None):
            batch_size_=states.shape[0]
        else:
            batch_size_=batch_size
        # print ("PPO p:", p)
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        advantage_tmp = advantage
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
        # r_ = np.mean(self._r(states, actions, 0))
        ### Give the metric some relative unit independent of action size
        if (("train_LSTM" in self._settings)
            and (self._settings["train_LSTM"] == True)):
            if ("train_LSTM_stateful" in self._settings
                and (self._settings["train_LSTM_stateful"] == True)
                # and False
                ):
                r_ = 0
        
        elif ("scale_r_by_action_size" in self.getSettings()
            and (self.getSettings()["scale_r_by_action_size"] == True)):
            r_ = ( 1 - np.mean(self._r([states, actions, 0])[0])) / float(self._action_length)
        else:
            r_ = ( 1 - np.mean(self._r([states, actions, 0])[0])) 
        # r_ = 0.98
        std = np.std(advantage)
        mean = np.mean(advantage)
        if ( 'advantage_scaling' in self.getSettings() and ( self.getSettings()['advantage_scaling'] != False) ):
            std = std / self.getSettings()['advantage_scaling']
            mean = 0.0
            print ("advantage_scaling: ", std)
        if ('normalize_advantage' in self.getSettings()
            and (self.getSettings()['normalize_advantage'] == True)):
            advantage = np.array((advantage - mean) / std, dtype=self._settings['float_type'])
        else:
            advantage = np.array((advantage / action_bound_std(self.getRewardBounds()) ) * (1.0-self.getSettings()['discount_factor']),
                                  dtype=self._settings['float_type'])
        ### check to not perform updates when r gets to large.
        if ('anneal_learning_rate' in self.getSettings()
            and (self.getSettings()['anneal_learning_rate'] == False)):
            pass
        else:
            K.set_value(self._model._actor_train.optimizer.lr, np.float32(self.getSettings()['learning_rate']) * p)
        et_factor = 0.2
        if ("ppo_et_factor" in self.getSettings()):
            et_factor = self.getSettings()["ppo_et_factor"] - 1.0
        if (r_ < (et_factor)) and ( r_ > (-et_factor)):  ### update not to large
            # lossActor = score.history['loss'][0]
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print ("Policy probability ratio: ", np.mean(r_))
            ### For now don't include dropout in policy updates
            if ("use_single_network" in self.getSettings() and ( self.getSettings()["use_single_network"] == True)):
                
                # (lossActor, r_) = self.trainPolicy([states, actions, result_states, rewards, advantage, p])
                if ('dont_use_td_learning' in self.getSettings() 
                and self.getSettings()['dont_use_td_learning'] == True):
                    if ( True ):
                        y_ = self._value_Target([result_states,0])[0]
                        target_ = rewards + ((self._discount_factor * y_))
                        target_2 = norm_reward(G_t, self.getRewardBounds()) * (1.0-self.getSettings()['discount_factor'])
                        target = (target_ + target_2) / 2.0
                    else:
                        target_ = norm_reward(G_t, self.getRewardBounds()) * (1.0-self.getSettings()['discount_factor'])
                    
                else:
                    y_ = self._value_Target([result_states,0])[0]
                    target_ = rewards + ((self._discount_factor * y_))
                target_ = np.array(target_, dtype=self._settings['float_type'])
                action_old = self._modelTarget.getActorNetwork().predict(states)
                ### Anneal learning rate
                self._model._actor_train.fit([states, action_old, advantage, (advantage * 0.0) + p], [actions, target_],
                      epochs=updates, batch_size=batch_size_,
                      verbose=0
                      )
            else: 
                if (("train_LSTM" in self._settings)
                    and (self._settings["train_LSTM"] == True)):
                    self.reset()
                    loss_ = []
                    if ("train_LSTM_stateful" in self._settings
                        and (self._settings["train_LSTM_stateful"] == True)
                        # and False
                        ):
                        
                        action_old = np.zeros((actions.shape))
                        for k in range(states.shape[1]):
                            x0 = np.array(states[:,[k]])
                            action_old__ = self._action_Target([x0, 0])
                            for j in range(states.shape[0]): 
                                action_old[j][k] = action_old__[0][j] ### Reducing dimensionality of targets
                        for k in range(states.shape[1]):
                            ### shaping data
                            states_ = np.array(states[:,[k]])
                            actions_old_ = np.array(action_old[:,k,:])
                            advantages_ = np.array(advantage[:,k,:])
                            actions_ = np.array(actions[:,k,:])
                            
                            self._model._actor_train.fit([states_, actions_old_, advantages_, (advantages_ * 0.0) + p], actions_,
                              epochs=1, 
                              batch_size=states.shape[0],
                              verbose=0,
                              )
                else:
                    ### Anneal learning rate
                    action_old = self._modelTarget.getActorNetwork().predict(states)
                    if ("policy_latent_length" in self.getSettings()):
                        action_old_next = self._modelTarget.getActorNetwork().predict(result_states)
                        self._model._actor_train.fit([states, action_old, advantage, (advantage * 0.0) + p,
                                                  result_states, action_old_next], actions,
                          epochs=updates, batch_size=batch_size_,
                          verbose=0,
                          )
                    else:
                        self._model._actor_train.fit([states, action_old, advantage, (advantage * 0.0) + p,
                                                  ], actions,
                          epochs=updates, batch_size=batch_size_,
                          verbose=0,
                          )
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print ("Policy loss: ", lossActor, " r: ", np.mean(r_))
            
            if ( (not np.isfinite(lossActor)) or (not np.isfinite(np.mean(r_)))):
                print("Something bad happend go back to the old policy")
                print ("State mean: ", np.mean(states, axis=0))
                print ("Actions mean: ", np.mean(actions, axis=0))
                print ("Advantage: ", advantage_tmp)
                self._model.getCriticNetwork().set_weights( copy.deepcopy(self._modelTarget.getCriticNetwork().get_weights()))
                self._model.getActorNetwork().set_weights( copy.deepcopy(self._modelTarget.getActorNetwork().get_weights()))
                if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                    r_ = np.mean(self._r([states, actions, 0])[0])
                    print ("Policy probability ratio: ", np.mean(r_))
                    print ("Policy mean: ", np.mean(self._policy_mean([states, 0])[0], axis=0))
                    print ("Policy std: ", np.mean(self._q_action_std([states, 0])[0], axis=0))
        else:
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print ("Policy Gradient too large: ", r_)
            
        return lossActor
    
    def get_actor_regularization(self):
        return self._get_actor_regularization([])
    
    def get_actor_loss(self, state, action, reward, nextState, advantage):
        anneal = (np.asarray(advantage) * 0.0) + 1.0
        if ("policy_latent_length" in self.getSettings()):
            return self._get_actor_loss([state, action, advantage, anneal, nextState])
        else:
            return self._get_actor_loss([state, action, advantage, anneal])
    
    def get_critic_regularization(self):
        return self._get_critic_regularization([])
    
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
        self._model._actor_train.save(fileName+"_actor_train"+suffix, overwrite=True, include_optimizer=True)
        self._model._actor.save(fileName+"_actor"+suffix, overwrite=True, include_optimizer=True)
        self._model._critic.save(fileName+"_critic"+suffix, overwrite=True, include_optimizer=True)
        self._modelTarget._actor.save(fileName+"_actor_T"+suffix, overwrite=True, include_optimizer=True)
        self._modelTarget._critic.save(fileName+"_critic_T"+suffix, overwrite=True, include_optimizer=True)
        try:
            from keras.utils import plot_model
            ### Save model design as image
            plot_model(self._model._actor, to_file=fileName+"_actor"+'.svg', show_shapes=True)
            plot_model(self._model._critic, to_file=fileName+"_critic"+'.svg', show_shapes=True)
            plot_model(self._model._actor_train, to_file=fileName+"_actor_train"+'.svg', show_shapes=True)
        except Exception as inst:
            ### Maybe the needed libraries are not available
            print ("Error saving diagrams for rl models.")
            print (inst)

    def loadFrom(self, fileName):
        from keras.models import load_model
        suffix = ".h5"
        print ("Loading agent: ", fileName)
        ### Because the simulation and learning use different model types (statefull vs stateless lstms...)
        actor = load_model(fileName+"_actor"+suffix)
        critic = load_model(fileName+"_critic"+suffix)
        self._model._actor.set_weights(actor.get_weights())
        # self._model._actor.optimizer = actor.optimizer
        self._model._critic.set_weights(critic.get_weights())
        self._model._critic.optimizer = critic.optimizer
        print ("self._Advantage: ", self._Advantage)
        ### TODO: fix this, still can't load training configuration
        """
        self._model._actor_train = load_model(fileName+"_actor_train"+suffix, 
                                              custom_objects={
                                                              'Advantage': self._Advantage
                                                              ,'loss': self._poli_loss(self._PoliAction,
                                                                                      self._Advantage, 
                                                                                      self._Anneal)
                                                              # ,'loss': self.__loss
                                                              })
        """
        if (self._modelTarget is not None):
            actor = load_model(fileName+"_actor_T"+suffix)
            critic = load_model(fileName+"_critic_T"+suffix)
            
            self._modelTarget._actor.set_weights(actor.get_weights())
            # self._modelTarget._actor.optimizer = actor.optimizer
            self._modelTarget._critic.set_weights(critic.get_weights())
            # self._modelTarget._critic.optimizer = critic.optimizer
        # self._model._actor_train = load_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        # self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        hf.close()
        
        PPO_KERAS.compile(self)
        self.updateTargetModel()
        

