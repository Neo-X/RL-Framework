## Nothing for now...

# from modular_rl import *

# ================================================================
# Trust Region Policy Optimization
# ================================================================


import theano
from theano import tensor as T
from collections import OrderedDict
import numpy as np
import sys
import copy
sys.path.append('../')
from model.ModelUtil import norm_state, scale_state, norm_action, scale_action, action_bound_std, scale_reward, randomExporationSTD
from model.LearningUtil import loglikelihood_keras, likelihood_keras, kl_keras, kl_D_keras, entropy_keras, flatgrad_keras, zipsame, get_params_flat, setFromFlat
from algorithm.KERASAlgorithm import *
import keras.backend as K
import keras
from keras.models import Sequential, Model


# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class TRPO_KERAS(KERASAlgorithm):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False):

        super(TRPO_KERAS,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        # create a small convolutional neural network
        
        # self._Anneal = keras.layers.Input(batch_shape=(1,), name="Anneal")
        self._Anneal = keras.layers.Input(shape=(1,), name="Anneal")
        # self._Anneal = K.variable(value=np.float32(1.0) ,name="Anneal")
        # self._Anneal = K.placeholder(ndim=0, name="Anneal")
        
        self._Advantage = keras.layers.Input(shape=(1,), name="Advantage")
        # self._Advantage = K.placeholder(shape=(1,), name="Advantage")
        
        self._PoliAction = keras.layers.Input(shape=(self._action_length,), name="PoliAction")
        if ( 'use_stochastic_policy' in self.getSettings() and ( self.getSettings()['use_stochastic_policy'])):
            self._PoliAction = keras.layers.Input(shape=(self._action_length*2,), name="PoliAction")
        
        
        self._model._actor = Model(inputs=self._model.getStateSymbolicVariable(), outputs=self._model._actor)
        if (print_info):
            print("Actor summary: ", self._model._actor.summary())
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
            
        """
        self._target_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype='float64'),
            broadcastable=(False, True))
        """
        self.__value = self._model.getCriticNetwork()([self._model.getStateSymbolicVariable()])
        self.__value_Target = self._modelTarget.getCriticNetwork()([self._model.getResultStateSymbolicVariable()])
        
        _target = self._model.getRewardSymbolicVariable() + (self._discount_factor * self.__value_Target)
        self._loss = K.mean(0.5 * (self.__value - _target) ** 2)
        
        
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
        
        self._actor_entropy = entropy_keras(self._q_valsActASTD)
        
        ## Compute on-policy policy gradient
        self._log_prob = loglikelihood_keras(self._model.getActionSymbolicVariable(), self._q_valsActA, self._q_valsActASTD, self._action_length)
        ### How should this work if the target network is very odd, as in not a slightly outdated copy.
        self._log_prob_target = loglikelihood_keras(self._model.getActionSymbolicVariable(), self._q_valsActTarget_State, self._q_valsActTargetSTD, self._action_length)
        ## This does the sum already
        self.__r = T.exp(self._log_prob - self._log_prob_target)
        # self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((self._r), self._Advantage)
        # self._actLoss_ = (self.__r) * self._Advantage
        self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)(T.exp(self._log_prob - self._log_prob_target), self._Advantage)
        ppo_epsilon = self.getSettings()['kl_divergence_threshold']
        # self._actLoss_2 = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((theano.tensor.clip(self._r, 1.0 - (ppo_epsilon * self._Anneal), 1+ (ppo_epsilon * self._Anneal)), self._Advantage))
        # self._actLoss_2 = (K.clip(self.__r, 1.0 - (ppo_epsilon * self._Anneal), 1 + (ppo_epsilon * self._Anneal)), self._Advantage)
        # self._actLoss_ = K.minimum(self._actLoss_, self._actLoss_2)
        # self._actLoss = ((T.mean(self._actLoss_) )) + -self._actor_regularization
        # self._actLoss = (-1.0 * (T.mean(self._actLoss_) + (self.getSettings()['std_entropy_weight'] * self._actor_entropy )))
        self._actLoss = -1.0 * K.mean(self._actLoss_)
        self._actLoss_tmp = self._actLoss
        if ("use_single_network" in self.getSettings() and ( self.getSettings()["use_single_network"] == True)):
            self._actLoss = self._actLoss + self._loss  
        
        # self._policy_grad = T.grad(self._actLoss ,  self._actionParams) 
        # self._actLoss_drop = (T.sum(0.5 * self._actDiff_drop ** 2)/float(self._batch_size)) # because the number of rows can shrink
        # self._actLoss_drop = (T.mean(0.5 * self._actDiff_drop ** 2))
        self._policy_grad = K.gradients(self._actLoss ,  self._model._actor.trainable_weights)
        self._kl_firstfixed = kl_keras(self._q_valsActTarget_State, self._q_valsActTargetSTD, self._q_valsActA, self._q_valsActASTD, self._action_length).mean()
        
        # N = self._model.getStateSymbolicVariable().shape[0]
        N = 1
        params = self._model._actor.trainable_weights
        surr = self._actLoss * (1.0/N)
        self.pg = flatgrad_keras(surr, params)

        prob_mean_fixed = theano.gradient.disconnected_grad(self._q_valsActA)
        prob_std_fixed = theano.gradient.disconnected_grad(self._q_valsActASTD)
        kl_firstfixed = kl_keras(prob_mean_fixed, prob_std_fixed, self._q_valsActA, self._q_valsActASTD, self._action_length).sum()/N
        grads = T.grad(kl_firstfixed, params)
        self.flat_tangent = T.vector(name="flat_tan")
        shapes = [var.get_value(borrow=True).shape for var in params]
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            tangents.append(K.reshape(self.flat_tangent[start:start+size], shape))
            start += size
        self.gvp = T.add(*[T.sum(g*tangent) for (g, tangent) in zipsame(grads, tangents)]) #pylint: disable=E1111
        # Fisher-vector product
        self.fvp = flatgrad_keras(self.gvp, params)
        
        self.ent = entropy_keras(self._q_valsActASTD).mean()
        self.kl = kl_keras(self._q_valsActTarget_State, self._q_valsActTargetSTD, self._q_valsActA, self._q_valsActASTD, self._action_length).mean()
        
        self.losses = [surr, self.kl, self.ent]
        self.loss_names = ["surr", "kl", "ent"]

        self.args = [self._model.getStateSymbolicVariable(), 
                     self._model.getActionSymbolicVariable(), 
                     self._Advantage
                     # self._q_valsActTarget_
                     ]
        
        
        self.args_fvp = [self._model.getStateSymbolicVariable(), 
                     # self._model.getActionSymbolicVariable()
                     # self._Advantage,
                     # self._q_valsActTarget_
                     ]
        
        self._givens_grad = [
            self._model.getStateSymbolicVariable()
            # self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            # self._model.getActionSymbolicVariable(): self._actions_shared,
        ]
        
        ## Bellman error
        # self._bellman = self._target - self._q_funcTarget
        
        TRPO_KERAS.compile(self)
        
    def compile(self):
        
        #### Stuff for Debugging #####
        #### Stuff for Debugging #####
        sgd = getOptimizer(lr=np.float32(self.getSettings()['critic_learning_rate']), 
                                    settings=self.getSettings())
        print ("Clipping: ", sgd.decay)
        print("sgd, critic: ", sgd)
        
        self._model.getCriticNetwork().compile(loss='mse', optimizer=sgd)
        # sgd = SGD(lr=0.0005, momentum=0.9)
        # self._get_advantage = theano.function([], [self._Advantage])
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
        # self._get_actor_diff_ = theano.function([], [self._actDiff], givens= self._actGivens)
        """{
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
            # self._Fallen: self._fallen_shared
        }) """
        
        
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
        
        self._get_log_prob = theano.function([self._model.getStateSymbolicVariable(),
                                              self._model.getActionSymbolicVariable()], self._log_prob)
        
        self._q_action_std = theano.function([self._model.getStateSymbolicVariable()], self._q_valsActASTD)
        # self._compute_fisher_vector_product = theano.function([flat_tangent] + args, fvp, **FNOPTS)
        self.kl_divergence = theano.function([self._model.getStateSymbolicVariable()], self._kl_firstfixed)
        
        self.compute_policy_gradient = theano.function(self.args, self.pg)
        self.compute_losses = theano.function(self.args, self.losses)
        self.compute_fisher_vector_product = theano.function([self.flat_tangent] + self.args_fvp, self.fvp)
        
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
    
    def setData(self, states, actions, rewards, result_states, fallen):
        self._model.setStates(states)
        self._model.setResultStates(result_states)
        self._model.setActions(actions)
        self._model.setRewards(rewards)
        self._modelTarget.setStates(states)
        self._modelTarget.setResultStates(result_states)
        self._modelTarget.setActions(actions)
        self._modelTarget.setRewards(rewards)
        # print ("Falls: ", fallen)
        # self._fallen_shared.set_value(fallen)
        # diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
    def trainActor(self, states, actions, rewards, result_states, falls, 
                   advantage, exp_actions=None, 
                   G_t=[[0]], forwardDynamicsModel=None, p=1.0, updates=1, batch_size=None):
        
        # if ('use_GAE' in self.getSettings() and ( self.getSettings()['use_GAE'] )):
            # self._advantage_shared.set_value(advantage)
            ## Need to scale the advantage by the discount to help keep things normalized
        std = np.std(advantage)
        mean = np.mean(advantage)
        if ( 'advantage_scaling' in self.getSettings() and ( self.getSettings()['advantage_scaling'] != False) ):
            std = std / self.getSettings()['advantage_scaling']
            mean = 0.0
            print ("advantage_scaling: ", std)
        if ('normalize_advantage' in self.getSettings()
            and (self.getSettings()['normalize_advantage'] == True)):
            # print("Normalize advantage")
            advantage = np.array((advantage - mean) / std, dtype=self._settings['float_type'])
        else:
            # print("Scale advantage")
            advantage = np.array((advantage / action_bound_std(self.getRewardBounds()) ) * (1.0-self.getSettings()['discount_factor']),
                                  dtype=self._settings['float_type'])
        # pass # use given advantage parameter
        self.setData(states, actions, rewards, result_states, falls)
        # advantage = self._get_advantage()[0] * (1.0/(1.0-self._discount_factor))
        # self._advantage_shared.set_value(advantage)
        #else:
        #    self.setData(states, actions, rewards, result_states, falls)
            # advantage = self._get_advantage()[0] * (1.0/(1.0-self._discount_factor))
        #    self._advantage_shared.set_value(advantage)
            
        ### Used to understand the shape of the parameters
        all_paramsActA = self._model.getActorNetwork().get_weights()
        self._modelTarget.getActorNetwork().set_weights( copy.deepcopy(self._model.getActorNetwork().get_weights()))
        # print ("Performing Critic trainning update")
        # if (( self._updates % self._weight_update_steps) == 0):
        #     self.updateTargetModel()
        # self._updates += 1
        # loss, _ = self._train()
        # print( "Actor loss: ", self._get_action_diff())
        lossActor = 0
        
        # diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        # print("Advantage: ", np.mean(self._get_advantage()))
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print("Rewards: ", np.mean(scale_reward(rewards, self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))), " std: ", np.std(scale_reward(rewards, self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))), " shape: ", np.array(rewards).shape)
            # print("Targets: ", np.mean(self._get_target()), " std: ", np.std(self._get_target()))
            print("Falls: ", np.mean(falls), " std: ", np.std(falls))
            # print("values, falls: ", np.concatenate((scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor'])), falls), axis=1))
            print("values: ", np.mean(scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))),
                   " std: ", np.std(scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))) )
            print("Model Advantage: ", np.mean(self._get_diff()), " std: ", np.std(self._get_diff()))
            
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Advantage: ", np.mean(advantage), " std: ", np.std(advantage))
            
            # print("Advantage, reward: ", np.concatenate((advantage, rewards), axis=1))
            print("Actions:     ", np.mean(actions, axis=0), " shape: ", actions.shape)
            print ("Policy mean: ", np.mean(self._policy_mean([states, 0])[0], axis=0))
            # print("Actions std:  ", np.mean(np.sqrt( (np.square(np.abs(actions - np.mean(actions, axis=0))))/1.0), axis=0) )
            # print("Actions std:  ", np.std(actions - self._q_action(), axis=0) )
            print("Actions std:  ", np.std(actions - self._policy_mean([states, 0])[0], axis=0) )
            print ("Policy std: ", np.mean(self.q_valsActASTD([states, 0])[0], axis=0))
            # print("Policy log prob before: ", np.mean(self._get_log_prob(states, actions), axis=0))
            # print( "Actor loss: ", np.mean(self._get_action_diff()))
            # print ("Actor diff: ", np.mean(np.array(self._get_diff()) / (1.0/(1.0-self._discount_factor))))
            ## Sometimes really HUGE losses appear, ocasionally
            # if (np.abs(np.mean(self._get_action_diff())) < 10): 
            #     lossActor, _ = self._trainActor()
            
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print("Policy   std2: ", np.mean(self._q_action_std(), axis=0) + np.std(self._q_action(), axis=0))
            new_actions = self._q_action()
            new_action_stds = self._q_action_std()
            new_actions_ = []
            for i in range(new_actions.shape[0]):
                action__ = randomExporationSTD(0.0, new_actions[i], new_action_stds[i])
                new_actions_.append(action__)
            print ("New action mean: ", np.mean(new_actions_, axis=0) )
            print ("New action std: ", np.std(new_actions_, axis=0) )
        
        self.getSettings()['cg_damping'] = 1e-3
        """
        cfg = self.cfg
        prob_np = concat([path["prob"] for path in paths])
        ob_no = concat([path["observation"] for path in paths])
        action_na = concat([path["action"] for path in paths])
        advantage_n = concat([path["advantage"] for path in paths])
        """
        args = (states, actions, advantage)
        args_fvp = (states)

        thprev = get_params_flat(self._model.getActorNetwork().get_weights())
        def fisher_vector_product(p):
            # print ("fvp p: ", p)
            # print ("states: ", p)
            # print ('cg_damping', self.getSettings()['cg_damping'] )
            fvp_ = self.compute_fisher_vector_product(p, states)+np.float32(self.getSettings()['cg_damping'])*p #pylint: disable=E1101,W0640
            # print ("fvp_ : ", fvp_)
            return fvp_
        g = self.compute_policy_gradient(*args)
        print ("g: ", g)
        losses_before = self.compute_losses(*args)
        if np.allclose(g, 0):
            print ("got zero gradient. not updating")
        else:
            stepdir = cg(fisher_vector_product, -g)
            # print ("stepdir: ", stepdir)
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            # print ("shs: ", shs )
            lm = np.sqrt(shs / np.float32(self.getSettings()['kl_divergence_threshold']))
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print ("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)
            def loss(th):
                # self.set_params_flat(th)
                params_tmp = setFromFlat(all_paramsActA, th)
                self._model.getActorNetwork().set_weights(params_tmp)
                return self.compute_losses(*args)[0] #pylint: disable=W0640
            success, theta = linesearch(loss, thprev, fullstep, neggdotstepdir/lm)
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print ("success", success)
            params_tmp = setFromFlat(all_paramsActA, theta)
            self._model.getActorNetwork().set_weights(params_tmp)
            # self.set_params_flat(theta)
        losses_after = self.compute_losses(*args)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Policy log prob after: ", np.mean(self._get_log_prob(states, actions), axis=0))

        out = OrderedDict()
        for (lname, lbefore, lafter) in zipsame(self.loss_names, losses_before, losses_after):
            out[lname+"_before"] = lbefore
            out[lname+"_after"] = lafter
    
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):    
            print( "Losses before: ", self.loss_names, ", ", losses_before)
            print( "Losses after: ", self.loss_names, ", ", losses_after)
        
        return out
        # print("Policy log prob after: ", np.mean(self._get_log_prob(), axis=0))
        # print( "Length of positive actions: " , str(len(tmp_actions)), " Actor loss: ", lossActor)
        # print( " Actor loss: ", lossActor)
        # self._advantage_shared.set_value(diff_)
        # lossActor, _ = self._trainActor()
        # kl_after = self.kl_divergence()
        """
        if kl_d > self.getSettings()['kl_divergence_threshold']:
            self._kl_weight_shared.set_value(self._kl_weight_shared.get_value()*2.0)
        else:
            self._kl_weight_shared.set_value(self._kl_weight_shared.get_value()/2.0)
        """  
    
        # return lossActor
    
    def train(self, states, actions, rewards, result_states, falls):
        loss = self.trainCritic(states, actions, rewards, result_states, falls)
        lossActor = self.trainActor(states, actions, rewards, result_states, falls)
        return loss
    
    def predict_std(self, state, deterministic_=True, p=1.0):
        # print ("PPO std p:", p)
        state = norm_state(state, self._state_bounds)   
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            action_std = self._q_action_std(state)
            # action_std = self._q_action_std()[0] * (action_bound_std(self._action_bounds))
        else:
            action_std = self._q_action_std(state) * (action_bound_std(self._action_bounds))
        return action_std

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
        
def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    
    # print "fval before", fval
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate*stepfrac
        ratio = actual_improve/expected_improve
        print ("a/e/r", actual_improve, expected_improve, ratio)
        if ratio > accept_ratio and actual_improve > 0:
            print ("fval after", newfval)
            return True, xnew
    return False, x

def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: 
        print (titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print (fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break
        
    if callback is not None:
        callback(x)
    if verbose: 
        print (fmtstr % (i+1, rdotr, np.linalg.norm(x))) # pylint: disable=W0631
    return x