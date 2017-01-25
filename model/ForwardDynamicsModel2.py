import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.AgentInterface import AgentInterface

class ForwardDynamicsNetwork(AgentInterface):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds):

        super(ForwardDynamicsNetwork,self).__init__(state_length, action_length, state_bounds, action_bounds, 0)
        
        batch_size=32
        self._next_state_length=6
        # data types for model
        State = T.dmatrix("State")
        State.tag.test_value = np.random.rand(batch_size,self._state_length)
        ResultState = T.dmatrix("ResultState")
        ResultState.tag.test_value = np.random.rand(batch_size,self._next_state_length)
        Action = T.dmatrix("Action")
        Action.tag.test_value = np.random.rand(batch_size, self._action_length)
        # create a small convolutional neural network
        inputLayerState = lasagne.layers.InputLayer((None, self._state_length), State)
        inputLayerAction = lasagne.layers.InputLayer((None, self._action_length), Action)
        concatLayer = lasagne.layers.ConcatLayer([inputLayerState, inputLayerAction])
        
        l_hid1ActA = lasagne.layers.DenseLayer(
                concatLayer, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
                
        l_hid2ActA = lasagne.layers.DenseLayer(
                l_hid1ActA, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3ActA = lasagne.layers.DenseLayer(
                l_hid2ActA, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid4ActA = lasagne.layers.DenseLayer(
                l_hid3ActA, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_out = lasagne.layers.DenseLayer(
                l_hid4ActA, num_units=6,
                nonlinearity=lasagne.nonlinearities.linear)
                # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._learning_rate = 0.001
        self._discount_factor= 0.8
        self._rho = 0.95
        self._rms_epsilon = 0.001
        
        self._updates=0
        
        self._states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((batch_size, self._next_state_length),
                     dtype=theano.config.floatX))

        self._actions_shared = theano.shared(
            np.zeros((batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
        inputs_ = {
            State: State,
            Action: Action,
        }
        self._forward = lasagne.layers.get_output(self._l_out, inputs_)
        
        # self._target = (Reward + self._discount_factor * self._q_valsB)
        self._diff = ResultState - self._forward
        self._loss = 0.5 * self._diff ** 2 + (1e-5 * lasagne.regularization.regularize_network_params(
                self._l_out, lasagne.regularization.l2))
        self._loss = T.mean(self._loss)
        
        self._params = lasagne.layers.helper.get_all_params(self._l_out)
        self._givens_ = {
            State: self._states_shared,
            ResultState: self._next_states_shared,
            Action: self._actions_shared,
        }
        
        # SGD update
        self._updates_ = lasagne.updates.rmsprop(self._loss, self._params, self._learning_rate, self._rho,
                                            self._rms_epsilon)
        # TD update
        # minimize Value function error
        #self._updates_ = lasagne.updates.rmsprop(T.mean(self._q_func) + (1e-4 * lasagne.regularization.regularize_network_params(
        #self._l_outA, lasagne.regularization.l2)), self._params, 
        #            self._learning_rate * -T.mean(self._diff), self._rho, self._rms_epsilon)
        
        
        # actDiff1 = (Action - self._q_valsActB) #TODO is this correct?
        # actDiff = (actDiff1 - (Action - self._q_valsActA))
        # actDiff = ((Action - self._q_valsActB2)) # Target network does not work well here?
        #self._actDiff = ((Action - self._q_valsActA)) # Target network does not work well here?
        #self._actLoss = 0.5 * self._actDiff ** 2 + (1e-4 * lasagne.regularization.regularize_network_params( self._l_outActA, lasagne.regularization.l2))
        #self._actLoss = T.mean(self._actLoss)
        
        
        
        
        self._train = theano.function([], [self._loss], updates=self._updates_, givens=self._givens_)
        self._forwardDynamics = theano.function([], self._forward,
                                       givens={State: self._states_shared, Action: self._actions_shared})
        inputs_ = [
                   State, 
                   ResultState,
                   Action
                   ]
        self._bellman_error = theano.function(inputs=inputs_, outputs=self._diff, allow_input_downcast=True)
        # self._diffs = theano.function(input=[State])
        
    def train(self, states, actions, result_states):
        self._states_shared.set_value(states)
        # extract only the columns needed to compute the reward
        self._next_states_shared.set_value(result_states[:,[3,4,5,6,7,8]])
        self._actions_shared.set_value(actions)
        # print ("Performing Critic trainning update")
        #if (( self._updates % self._weight_update_steps) == 0):
        #    self.updateTargetModel()
        self._updates += 1
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        loss = self._train()
        # This undoes the Actor parameter updates as a result of the Critic update.
        #if all_paramsActA == self._l_outActA:
        #    print ("Parameters the same:")
        # lasagne.layers.helper.set_all_param_values(self._l_outActA, all_paramsActA)
        # self._trainOneActions(states, actions, rewards, result_states)
        # diff_ = self._bellman_error(states, rewards, result_states)
        # print ("Diff")
        # print (diff_)
        return loss
    
    def predict(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = [norm_state(state, self._state_bounds)]
        action = [norm_action(action, self._action_bounds)]
        self._states_shared.set_value(state)
        self._actions_shared.set_value(action)
        state_ = scale_state(self._forwardDynamics()[0], self._state_bounds)
        return state_

    def bellman_error(self, state, action, result_state):
        # return self._bellman_error(state, reward, result_state)
        return self._bellman_error(state, result_state, action)
