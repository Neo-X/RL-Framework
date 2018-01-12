import theano
from theano import tensor as T
from lasagne.layers import get_all_params
import numpy as np
# import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface
from model.LearningUtil import loglikelihood, kl, entropy, change_penalty
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras


# For debugging
# theano.config.mode='FAST_COMPILE'
from collections import OrderedDict

class DPGKeras(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):
        """
            In order to get this to work we need to be careful not to update the actor parameters
            when updating the critic. This can be an issue when the Concatenating networks together.
            The first first network becomes a part of the second. However you can still access the first
            network by itself but an updates on the second network will effect the first network.
            Care needs to be taken to make sure only the parameters of the second network are updated.
        """
        
        super(DPGKeras,self).__init__( model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)

        self._modelTarget = copy.deepcopy(model)
        
            
        # print ("Initial W " + str(self._w_o.get_value()) )
        
            ## TD update
        DPGKeras.compile(self)
        
    def compile(self):
        
        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0)
        print ("Clipping: ", sgd.decay)
        self._model.getCriticNetwork().compile(loss='mse', optimizer=sgd)
        
        
        # sgd = SGD(lr=0.0005, momentum=0.9)
        ### loss function for actor, increase the Q value
        def neg_y(true_y, pred_y):
            return -pred_y
        self._actor_optimizer = keras.optimizers.Adam(lr=self.getSettings()['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0)
        print ("Clipping: ", sgd.decay)
        # Train function
        # updates = opt.get_updates(self._model.getActorNetwork().trainable_weights, [], loss)
        
        weights = [self._model._actionInput] + self._model.getCriticNetwork().trainable_weights # weight tensors
        # weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable] # filter down weights tensors to only ones which are trainable
        gradients = self._model.getCriticNetwork().optimizer.get_gradients(self._model.getCriticNetwork().total_loss, weights) # gradient tensors
        
        input_tensors = [self._model.getCriticNetwork().inputs[0], # input data
                         self._model.getCriticNetwork().inputs[1], # input data
                         self._model.getCriticNetwork().sample_weights[0], # how much to weight each sample by
                         self._model.getCriticNetwork().targets[0], # labels
                         K.learning_phase(), # train or test mode
        ]
        
        self._get_gradients = K.function(inputs=input_tensors, outputs=gradients)
        
        updates = self._actor_optimizer.get_updates(self._model.getActorNetwork().trainable_weights, loss=gradients, constraints=[])
        self._train = K.function([x, ytrue],[loss, accuracy],updates=updates)
        self._model.getActorNetwork().compile(loss=neg_y, optimizer=self._actor_optimizer)
        
        
        
    def getGrads(self, states, actions=None, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if ( alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=theano.config.floatX)
        self._model.setStates(states)
        if ( actions is None ):
            actions = self.predict_batch(states)
        self._model.setActions(actions)
        return self._get_state_grad()
    
    def getActionGrads(self, states, actions=None, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        
        if ( alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=theano.config.floatX)
        self._model.setStates(states)
        if ( actions is None ):
            actions = self.predict_batch(states)
        self._model.setActions(actions)
        
        input = [
              states, # X
              np.ones(states.shape[0]), # sample weights
              actions, # y
              0 # learning phase in TEST mode
        ]
        return self.self._get_gradients(input)
    
    def updateTargetModel(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Updating target Model")
        """
            Target model updates
        """
        # return
        ## I guess it is okay to lerp the entire network even though we only really want to 
        ## lerp the value function part of the networks, the target policy is not used for anythings
        all_paramsA = self._model.getCriticNetwork().get_weights()
        all_paramsB = self._modelTarget.getCriticNetwork().get_weights()
        lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        
        all_params = []
        for paramsA, paramsB in zip(all_paramsA, all_paramsB):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        self._modelTarget.getCriticNetwork().set_weights(all_params)
    
    def updateTargetModelValue(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Updating MBAE target Model")
        """
            Target model updates
        """
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._model._value_function)
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork())
        lasagne.layers.helper.set_all_param_values(self._modelTarget._value_function, all_paramsA)
        # lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), all_paramsActA)
            
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
            params[0][i] = np.array(params[0][i], dtype=theano.config.floatX)
            """
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
        self._fallen_shared.set_value(fallen)
        # diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        ## Easy fix for computing actor loss
        # diff = self._bellman_error2()
        # self._tmp_diff_shared.set_value(diff)
        
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
    def trainCritic(self, states, actions, rewards, result_states, falls):
        
        # self.setData(states, actions, rewards, result_states, falls)
        ## get actions for target policy
        target_actions = self._modelTarget.getActorNetwork().predict(states)
        ## Get next q value
        q_vals_b = self._modelTarget.getCriticNetwork().predict(states, actions)
        # q_vals_b = self._q_val()
        ## Compute target values
        # target_tmp_ = rewards + ((self._discount_factor* q_vals_b )* falls)
        target_tmp_ = rewards + ((self._discount_factor * q_vals_b ))
        # self.setData(states, actions, rewards, result_states, falls)
        # self._tmp_target_shared.set_value(target_tmp_)
        
        # self._target = T.mul(T.add(self._model.getRewardSymbolicVariable(), T.mul(self._discount_factor, self._q_valsB )), self._Fallen)
        
        loss = self.fit([states, actions], target_tmp_,
                        batch_size=32,
                        nb_epoch=1,
                        verbose=False,
                        shuffle=False)
        return loss
        
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions, forwardDynamicsModel=None):
        # self.setData(states, actions, rewards, result_states, falls)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print("values: ", np.mean(self._q_val()* (1.0 / (1.0- self.getSettings()['discount_factor']))), " std: ", np.std(self._q_val()* (1.0 / (1.0- self.getSettings()['discount_factor']))) )
            print("Rewards: ", np.mean(rewards), " std: ", np.std(rewards), " shape: ", np.array(rewards).shape)
        # print("Policy mean: ", np.mean(self._q_action(), axis=0))
        loss = 0
        # loss = self._trainActor()
        # print("******** Not learning actor right now *****")
        # return loss
        actions = self.getActorNetwork().predict(states, batch_size=states.shape[0])
        # print ("actions shape:", actions.shape)
        # next_states = forwardDynamicsModel.predict_batch(states, actions)
        # print ("next_states shape: ", next_states.shape)
        action_grads = self.getActionGrads(states, actions, alreadyNormed=True)[0]
        # print ("next_state_grads shape: ", next_state_grads.shape)
        # action_grads = forwardDynamicsModel.getGrads(states, actions, next_states, v_grad=next_state_grads, alreadyNormed=True)[0] * 1.0
        # print ( "action_grads shape: ", action_grads.shape)
        """
            From DEEP REINFORCEMENT LEARNING IN PARAMETERIZED ACTION SPACE
            Hausknecht, Matthew and Stone, Peter
            
            actions.shape == action_grads.shape
        """
        use_parameter_grad_inversion=True
        if ( use_parameter_grad_inversion ):
            for i in range(action_grads.shape[0]):
                for j in range(action_grads.shape[1]):
                    if (action_grads[i,j] > 0):
                        inversion = (1.0 - actions[i,j]) / 2.0
                    else:
                        inversion = ( actions[i,j] - (-1.0)) / 2.0
                    action_grads[i,j] = action_grads[i,j] * inversion
                    
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            # print("Actions mean:     ", np.mean(actions, axis=0))
            print("Policy mean: ", np.mean(self._q_action(), axis=0))
            # print("Actions std:  ", np.mean(np.sqrt( (np.square(np.abs(actions - np.mean(actions, axis=0))))/1.0), axis=0) )
            # print("Actions std:  ", np.std((actions - self._q_action()), axis=0) )
            # print("Actions std:  ", np.std((actions), axis=0) )
            # print("Policy std: ", np.mean(self._q_action_std(), axis=0))
            # print("Mean Next State Grad grad: ", np.mean(next_state_grads, axis=0), " std ", np.std(next_state_grads, axis=0))
            print("Mean action grad: ", np.mean(action_grads, axis=0), " std ", np.std(action_grads, axis=0))
        
        
        
        ## Why the -1.0??
        ## Because the SGD method is always performing MINIMIZATION!!
        self._action_grad_shared.set_value(-1.0*action_grads)
        self._trainActionGRAD()
        
        return loss
        
    def train(self, states, actions, rewards, result_states):
        loss = self.trainCritic(states, actions, rewards, result_states)
        lossActor = self.trainActor(states, actions, rewards, result_states)
        return loss
    
    def q_value(self, state):
        """
            For returning a vector of q values, state should NOT be normalized
        """
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        """
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            pass
        else:
        """
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        action = self._q_action()
        self._model.setActions(action)
        self._modelTarget.setActions(action)
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            return scale_reward(self._q_val(), self.getRewardBounds())[0] * (1.0 / (1.0- self.getSettings()['discount_factor']))
            # return (self._q_val())[0]
        else:
            return scale_reward(self._q_val(), self.getRewardBounds())[0] * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # return self._q_valTarget()[0]
        # return self._q_val()[0]
    
    def q_values(self, state):
        """
            For returning a vector of q values, state should already be normalized
        """
        """
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            pass
        else:
        """
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        action = self._q_action()
        self._model.setActions(action)
        self._modelTarget.setActions(action)
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            return scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
            # return (self._q_val())[0] 
        else:
            return scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # return self._q_valTarget()
        # return self._q_val()