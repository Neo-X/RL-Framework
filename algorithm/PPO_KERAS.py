import numpy as np
# import lasagne
import h5py
import sys
import copy
sys.path.append('../')
from model.ModelUtil import norm_state, scale_state, norm_action, scale_action, action_bound_std, scale_reward, norm_reward
from algorithm.KERASAlgorithm import KERASAlgorithm
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
    
def getOptimizer(lr, settings):
    """
        Function to make it easier to select the SGD optimizer to use
    """
    if ( "optimizer" in settings 
         and ( settings["optimizer"] == "sgd")):
        sgd = keras.optimizers.SGD(lr=lr, momentum=settings["rho"], decay=0.0, nesterov=False)
    else:
        sgd = keras.optimizers.Adam(lr=np.float32(lr), 
                                beta_1=settings["rms_epsilon"], beta_2=np.float32(0.999), 
                                epsilon=np.float32(settings["rms_epsilon"]), decay=0.0,
                                amsgrad=False)
    return sgd
    

class PPO_KERAS(KERASAlgorithm):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False):

        super(PPO_KERAS,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=print_info)
        
        print ("PPO_KERAS: created models")
        # self._modelTarget = model
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        # self._Anneal = keras.layers.Input(batch_shape=(1,), name="Anneal")
        self._Anneal = keras.layers.Input(shape=(1,), name="Anneal")
        # self._Anneal = K.variable(value=np.float32(1.0) ,name="Anneal")
        # self._Anneal = K.placeholder(ndim=0, name="Anneal")
        
        self._Advantage = keras.layers.Input(shape=(1,), name="Advantage")
        # self._Advantage = K.placeholder(shape=(1,), name="Advantage")
        
        self._PoliAction = keras.layers.Input(shape=(self._action_length,), name="PoliAction")
        if ( 'use_stocastic_policy' in self.getSettings() and ( self.getSettings()['use_stocastic_policy'])):
            self._PoliAction = keras.layers.Input(shape=(self._action_length*2,), name="PoliAction")
        
        ## primary network
        self._model = model
        input_ = [self._model.getStateSymbolicVariable(),
                 self._PoliAction,
                 self._Advantage,
                 self._Anneal
                  ]
        
        if ("use_single_network" in self.getSettings() and ( self.getSettings()["use_single_network"] == True)):
            self._model._actor_train = Model(inputs=input_, outputs=[self._model._actor, self._model._critic])
        else:
            self._model._actor_train = Model(inputs=input_, outputs=self._model._actor)
        self._model._actor = Model(inputs=self._model.getStateSymbolicVariable(), outputs=self._model._actor)
        if (print_info):
            print("Actor summary: ", self._model._actor_train.summary())
        self._model._critic = Model(inputs=self._model.getStateSymbolicVariable(), outputs=self._model._critic)
        if (print_info):
            print("Critic summary: ", self._model._critic.summary())
        ## Target network
        # self._modelTarget = copy.deepcopy(model)
        self._modelTarget = type(self._model)(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=print_info)
        input_Target = [self._modelTarget.getStateSymbolicVariable(),
                 self._PoliAction,
                 self._Advantage,
                 self._Anneal
                  ]
        self._modelTarget._actor = Model(inputs=self._modelTarget.getStateSymbolicVariable(), outputs=self._modelTarget._actor)
        if (print_info):
            print("Target Actor summary: ", self._modelTarget._actor.summary())
        self._modelTarget._critic = Model(inputs=self._modelTarget.getStateSymbolicVariable(), outputs=self._modelTarget._critic)
        if (print_info):
            print("Target Critic summary: ", self._modelTarget._critic.summary())
        
        self.__value = self._model.getCriticNetwork()([self._model.getStateSymbolicVariable()])
        self.__value_Target = self._modelTarget.getCriticNetwork()([self._model.getResultStateSymbolicVariable()])
        
        _target = self._model.getRewardSymbolicVariable() + (self._discount_factor * self.__value_Target)
        self._loss = K.mean(0.5 * (self.__value - _target) ** 2)
        
        
        self._q_valsActA = self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,:self._action_length]
        if ( 'use_stocastic_policy' in self.getSettings() and ( self.getSettings()['use_stocastic_policy'])): 
            # self._q_valsActASTD = (self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) + 1e-2
            self._q_valsActASTD = ((self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) * self.getSettings()['exploration_rate']) + 1e-2
        else:
            self._q_valsActASTD = ( K.ones_like(self._q_valsActA)) * self.getSettings()['exploration_rate']
        
        self._q_valsActTarget_State = self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,:self._action_length]
        if ( 'use_stocastic_policy' in self.getSettings() and ( self.getSettings()['use_stocastic_policy'])): 
            # self._q_valsActTargetSTD = (self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]) + 1e-2
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
        # self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((self._r), self._Advantage)
        self._actLoss_ = (self.__r) * self._Advantage
        ppo_epsilon = self.getSettings()['kl_divergence_threshold']
        # self._actLoss_2 = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((theano.tensor.clip(self._r, 1.0 - (ppo_epsilon * self._Anneal), 1+ (ppo_epsilon * self._Anneal)), self._Advantage))
        self._actLoss_2 = (K.clip(self.__r, 1.0 - (ppo_epsilon * self._Anneal), 1 + (ppo_epsilon * self._Anneal)), self._Advantage)
        self._actLoss_ = K.minimum(self._actLoss_, self._actLoss_2)
        # self._actLoss = ((T.mean(self._actLoss_) )) + -self._actor_regularization
        # self._actLoss = (-1.0 * (T.mean(self._actLoss_) + (self.getSettings()['std_entropy_weight'] * self._actor_entropy )))
        self._actLoss = -1.0 * K.mean(self._actLoss_)
        self._actLoss_tmp = self._actLoss
        if ("use_single_network" in self.getSettings() and ( self.getSettings()["use_single_network"] == True)):
            self._actLoss = self._actLoss + self._loss  
        
        # self._policy_grad = T.grad(self._actLoss ,  self._actionParams)
        
        PPO_KERAS.compile(self)
        self.updateTargetModel()
        
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
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
        
        def poli_loss(action_old, advantage, anneal):
            ## Compute on-policy policy gradient
            action_old_mean = action_old[:,:self._action_length]
            if ( 'use_stocastic_policy' in self.getSettings() and ( self.getSettings()['use_stocastic_policy'])):
                action_old_std = action_old[:,self._action_length:]
            else:
                action_old_std = (K.ones_like(action_old_mean)) * self.getSettings()['exploration_rate']
            
            def loss(action_true, action_pred):
                action_true = action_true[:,:self._action_length]
                action_pred_mean = action_pred[:,:self._action_length]
                if ( 'use_stocastic_policy' in self.getSettings() and ( self.getSettings()['use_stocastic_policy'])):
                    action_pred_std = action_pred[:,self._action_length:]
                else:
                    action_pred_std = (K.ones_like(action_pred_mean)) * self.getSettings()['exploration_rate']
                prob = likelihood_keras(action_true, action_pred_mean, action_pred_std, self._action_length)
                prob_target = likelihood_keras(action_true, action_old_mean, action_old_std, self._action_length)
                _r = (prob / prob_target)
                actLoss_ = (_r) * advantage
                ppo_epsilon = self.getSettings()['kl_divergence_threshold']
                actLoss_2 = (K.clip(_r, 1.0 - (ppo_epsilon * anneal), 1 + (ppo_epsilon * anneal)), advantage)
                actLoss_ = K.minimum(actLoss_, actLoss_2)
                ### Average across action dimensions
                actLoss = -1.0 * K.mean(actLoss_, axis=-1)
                ### Average over batch
                return K.mean(actLoss)
            
            return loss
        
        
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
        self._get_actor_loss = K.function([self._model.getStateSymbolicVariable(),
                                                 self._model.getActionSymbolicVariable(),
                                                 self._Advantage,
                                                 self._Anneal  
                                                 # ,K.learning_phase()
                                                 ], [self._actLoss_tmp])
        print ("build actor updates")
        """
        if ("use_single_network" in self.getSettings() and ( self.getSettings()["use_single_network"] == True)):
            ### Hck for now until I properly support keras optimizers
            poli_updates = adam_updates(self._actLoss + self._critic_regularization, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate * self._Anneal)
            if ("learning_backend" in self.getSettings() and (self.getSettings()["learning_backend"] == "tensorflow")):
                poli_updates = list(poli_updates)
                # print ("poli_updates: ", poli_updates)
            else:
                poli_updates = poli_updates.items()
            self.trainPolicy = K.function([self._model.getStateSymbolicVariable(),
                                                 self._model.getActionSymbolicVariable(),
                                                 self._model.getResultStateSymbolicVariable(),
                                                 self._model.getRewardSymbolicVariable(),
                                                 self._Advantage,
                                                 self._Anneal  
                                                 # ,K.learning_phase()
                                                 ], [self._actLoss, self.__r], 
                            updates= poli_updates
                            # updates= adam_updates(self._actLoss, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate * self._Anneal).items()
                            # ,on_unused_input='warn'
                            # updates= adam_updates(self._actLoss, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate).items()
                            )
        else:
            poli_updates = updates= adam_updates(self._actLoss + self._actor_regularization, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate * self._Anneal)
            if ("learning_backend" in self.getSettings() and (self.getSettings()["learning_backend"] == "tensorflow")):
                poli_updates = list(poli_updates)
                print ("poli_updates: ", poli_updates)
            else:
                poli_updates = poli_updates.items()
            print("poli_updates: ", poli_updates)
            self.trainPolicy = K.function([self._model.getStateSymbolicVariable(),
                                                 self._model.getActionSymbolicVariable(),
                                                 self._Advantage,
                                                 self._Anneal  
                                                 # ,K.learning_phase()
                                                 ], [self._actLoss, self.__r], 
                            updates= poli_updates
                            # updates= adam_updates(self._actLoss, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate * self._Anneal).items()
                            # ,on_unused_input='warn'
                            # updates= adam_updates(self._actLoss, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate).items()
                            )
            self._get_actor_loss = K.function(
                                                   [self._model.getStateSymbolicVariable(),
                                                 self._model.getActionSymbolicVariable(),
                                                 self._Advantage,
                                                 self._Anneal  
                                                 # ,K.learning_phase()
                                                 ],
                                                 [self._actLoss]
                                                 # ,on_unused_input='warn'
                                                 )
        """
        
        self._r = K.function([self._model.getStateSymbolicVariable(),
                                     self._model.getActionSymbolicVariable()
                                     # ,self._Anneal
                                     # ,K.learning_phase()
                                     ], 
                                  [self.__r]
                                  # ,on_unused_input='warn'
                                  )
        
        gradients = K.gradients(K.mean(self.__value), [self._model.getStateSymbolicVariable()]) # gradient tensors
        self._get_gradients = K.function(inputs=[self._model.getStateSymbolicVariable(),  K.learning_phase()], outputs=gradients)
        
        self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        if ("use_target_net_for_critic" in self.getSettings() and
            (self.getSettings()["use_target_net_for_critic"] == False)):
            self._value_Target = self._value
        else:
            self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        
        self._policy_mean = K.function([self._model.getStateSymbolicVariable(), 
                                          K.learning_phase()], [self._q_valsActA])
        self.q_valsActASTD = K.function([self._model.getStateSymbolicVariable(), 
                                          # self._Anneal,
                                          K.learning_phase()], [self._q_valsActASTD]) 
        
        print ("Done building PPO KERAS")

    """     
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
            if ( True ):
                y_ = self._value_Target([result_states,0])[0]
                # v = self._model.getCriticNetwork().predict(states, batch_size=states.shape[0])
                # target_ = rewards + ((self._discount_factor * y_) * falls)
                target_ = rewards + ((self._discount_factor * y_))
                target_2 = norm_reward(G_t, self.getRewardBounds()) * (1.0-self.getSettings()['discount_factor'])
                target = (target_ + target_2) / 2.0
            else:
                target_ = norm_reward(G_t, self.getRewardBounds()) * (1.0-self.getSettings()['discount_factor'])
        else:
            # y_ = self._modelTarget.getCriticNetwork().predict(result_states, batch_size=states.shape[0])
            y_ = self._value_Target([result_states,0])[0]
            # v = self._model.getCriticNetwork().predict(states, batch_size=states.shape[0])
            # target_ = rewards + ((self._discount_factor * y_) * falls)
            target_ = rewards + ((self._discount_factor * y_))
        target_ = np.array(target_, dtype=self._settings['float_type'])
        # states = np.array(states, dtype=self._settings['float_type'])
        # print ("target type: ", target_.dtype)
        # print ("states type: ", states.dtype)
        v = self.q_values(states)
        v_ = self._value_Target([states,0])[0]
        print ("Critic Target: ", np.concatenate((v, y_, v - y_, v - v_, rewards, target_) ,axis=1) )
        c_error = np.mean(np.mean(np.square(v - target_), axis=1))
        # print ("critic error: ", np.mean(np.mean(np.square(v - target_), axis=1)))
        if (c_error < 0.25):
            score = self._model.getCriticNetwork().fit(states, target_,
                  epochs=updates, batch_size=batch_size_,
                  verbose=0
                  # callbacks=[early_stopping],
                  )
            loss = score.history['loss'][0]
        else:
            print ("Critic error to high:", c_error)
            loss = 0
        # print(" Critic loss: ", loss)
        
        return loss
"""    
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions=None, 
                   G_t=[[0]], forwardDynamicsModel=None, p=1.0, updates=1, batch_size=None):
        lossActor = 0
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
        r_ = ( 1 - np.mean(self._r([states, actions])[0])) / float(self._action_length)
        # r_ = 0.98
        std = np.std(advantage)
        mean = np.mean(advantage)
        if ( 'advantage_scaling' in self.getSettings() and ( self.getSettings()['advantage_scaling'] != False) ):
            std = std / self.getSettings()['advantage_scaling']
            mean = 0.0
        if ('normalize_advantage' in self.getSettings()
            and (self.getSettings()['normalize_advantage'] == True)):
            # print("Normalize advantage")
            advantage = np.array((advantage - mean) / std, dtype=self._settings['float_type'])
        else:
            # print("Scale advantage")
            advantage = np.array((advantage / action_bound_std(self.getRewardBounds()) ) * (1.0-self.getSettings()['discount_factor']),
                                  dtype=self._settings['float_type'])
            # print ("advantage mean, std ", np.mean(advantage), np.std(advantage) )
        ### check to not perform updates when r gets to large.
        if ('anneal_learning_rate' in self.getSettings()
            and (self.getSettings()['anneal_learning_rate'] == False)):
            pass
        else:
            K.set_value(self._model._actor_train.optimizer.lr, np.float32(self.getSettings()['learning_rate']) * p)
        et_factor = 0.2
        if ("ppo_et_factor" in self.getSettings()):
            et_factor = self.getSettings()["ppo_et_factor"] - 1.0
            # print("Updated et_factor: ", et_factor)
        if (r_ < (et_factor)) and ( r_ > (-et_factor)):  ### update not to large
            # lossActor = score.history['loss'][0]
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print ("Policy probability ratio: ", np.mean(r_))
                print ("Policy mean: ", np.mean(self._policy_mean([states, 0])[0], axis=0))
                print ("Policy std: ", np.mean(self.q_valsActASTD([states, 0])[0], axis=0))
                # print ("Network Params mean: ", np.mean(np.array(list(flatten(self.getNetworkParameters()[1])))))
                # print ("States shape: ", np.array(states).shape)
                        ### For now don't include dropout in policy updates
            if ("use_single_network" in self.getSettings() and ( self.getSettings()["use_single_network"] == True)):
                
                # (lossActor, r_) = self.trainPolicy([states, actions, result_states, rewards, advantage, p])
                if ('dont_use_td_learning' in self.getSettings() 
                and self.getSettings()['dont_use_td_learning'] == True):
                    if ( True ):
                        y_ = self._value_Target([result_states,0])[0]
                        # v = self._model.getCriticNetwork().predict(states, batch_size=states.shape[0])
                        # target_ = rewards + ((self._discount_factor * y_) * falls)
                        target_ = rewards + ((self._discount_factor * y_))
                        target_2 = norm_reward(G_t, self.getRewardBounds()) * (1.0-self.getSettings()['discount_factor'])
                        target = (target_ + target_2) / 2.0
                    else:
                        target_ = norm_reward(G_t, self.getRewardBounds()) * (1.0-self.getSettings()['discount_factor'])
                    
                else:
                    y_ = self._value_Target([result_states,0])[0]
                    # v = self._model.getCriticNetwork().predict(states, batch_size=states.shape[0])
                    # target_ = rewards + ((self._discount_factor * y_) * falls)
                    target_ = rewards + ((self._discount_factor * y_))
                target_ = np.array(target_, dtype=self._settings['float_type'])
                action_old = self._modelTarget.getActorNetwork().predict(states)
                ### Anneal learning rate
                # print ("model learning rate: ", K.get_value(self._model._actor_train.optimizer.lr))
                self._model._actor_train.fit([states, action_old, advantage, (advantage * 0.0) + p], [actions, target_],
                      epochs=updates, batch_size=batch_size_,
                      verbose=0,
                      shuffle=True
                      # p_=p,
                      # callbacks=self._callbacks_list,
                      )
            else: 
                # (lossActor, r_) = self.trainPolicy(states, actions, advantage, p, 0)
                # print("states: ", states)
                # action_old = self._modelTarget.getActorNetwork().predict([states, actions, advantage, advantage])[:,:self._action_length]
                action_old = self._modelTarget.getActorNetwork().predict(states)
                ### Anneal learning rate
                self._model._actor_train.fit([states, action_old, advantage, (advantage * 0.0) + p], actions,
                      epochs=updates, batch_size=batch_size_,
                      verbose=0,
                      shuffle=True
                      # callbacks=[early_stopping],
                      )
                # (lossActor, r_) = self.trainPolicy([states, actions, advantage, p])
                """
                score = self._model.getActorNetwork().fit([states, actions, advantage], advantage,
                          nb_epoch=1, batch_size=32,
                          verbose=0
                          # callbacks=[early_stopping],
                          )
                """
            # (lossActor, r_) = self.trainPolicy(states, actions, advantage, 1.0)
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                r_ = np.mean(self._r([states, actions])[0])
                print ("Policy loss: ", lossActor, " r: ", np.mean(r_))
            
            if ( (not np.isfinite(lossActor)) or (not np.isfinite(np.mean(r_)))):
                print("Something bad happend go back to the old policy")
                print ("State mean: ", np.mean(states, axis=0))
                print ("Actions mean: ", np.mean(actions, axis=0))
                print ("Advantage: ", advantage_tmp)
                # print ("Network Params mean: ", np.mean(np.array(list(flatten(self.getNetworkParameters()[1])))))
                self._model.getCriticNetwork().set_weights( copy.deepcopy(self._modelTarget.getCriticNetwork().get_weights()))
                self._model.getActorNetwork().set_weights( copy.deepcopy(self._modelTarget.getActorNetwork().get_weights()))
                if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                    r_ = np.mean(self._r([states, actions])[0])
                    print ("Policy probability ratio: ", np.mean(r_))
                    print ("Policy mean: ", np.mean(self._policy_mean([states, 0])[0], axis=0))
                    print ("Policy std: ", np.mean(self.q_valsActASTD([states, 0])[0], axis=0))
                    # print ("Network Params mean: ", np.mean(np.array(self.getNetworkParameters()[1])))
        else:
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print ("Policy Gradient too large: ", r_)
            
        return lossActor
    
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
        
    def get_actor_regularization(self):
        return self._get_actor_regularization([])
    
    def get_actor_loss(self, state, action, reward, nextState, advantage):
        anneal = (np.asarray(advantage) * 0.0) + 1.0
        return self._get_actor_loss([state, action, advantage, anneal])
    
    def get_critic_regularization(self):
        return self._get_critic_regularization([])
    
    def get_critic_loss(self, state, action, reward, nextState):
        return self._get_critic_loss([state, reward, nextState, 0])
    
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
        model_actor_train = self._model._actor_train
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
        self._model._actor_train = model_actor_train
        self._model._actor = model_actor
        self._model._critic = model_critic
        self._modelTarget._actor = modelTarget_actor
        self._modelTarget._critic = modelTarget_critic
        # print ("self._model._actor_train: ", self._model._actor_train)