import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface


# For debugging
# theano.config.mode='FAST_COMPILE'
from collections import OrderedDict

def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """RMSProp updates
    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients. See [1]_ for further description.
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    rho : float or symbolic scalar
        Gradient moving average decay factor
    epsilon : float or symbolic scalar
        Small value added for numerical stability
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.
    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:
    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}
    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """
    clip = 2.0
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    # grads = theano.gradient.grad_clip(grads, -clip, clip) 
    grads_ = []
    for grad in grads:
        grads_.append(theano.gradient.grad_clip(grad, -clip, clip) )
    grads = grads_
    
    print ("Grad Update: " + str(grads[0]) )
    
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates

class DPG(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):
        """
            In order to get this to work we need to be careful not to update the actor parameters
            when updating the critic. This can be an issue when the Concatenating networks together.
            The first first network becomes a part of the second. However you can still access the first
            network by itself but an updates on the second network will effect the first network.
            Care needs to be taken to make sure only the parameters of the second network are updated.
        """
        
        super(DPG,self).__init__( model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)

        self._Fallen = T.bcol("Fallen")
        ## because float64 <= float32 * int32, need to use int16 or int8
        self._Fallen.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype('int8'))
        
        self._fallen_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype='int8'),
            broadcastable=(False, True))
        
        self._modelTarget = copy.deepcopy(model)
        
            
        # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._weight_update_steps=self.getSettings()['steps_until_target_network_update']
        self._updates=0
        self._decay_weight=self.getSettings()['regularization_weight']
        self._critic_regularization_weight = self.getSettings()["critic_regularization_weight"]
        self._critic_learning_rate = self.getSettings()["critic_learning_rate"]
        
        self._q_valsA = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._q_valsA_drop = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        self._q_valsNextState = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
        self._q_valsTargetNextState = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
        self._q_valsTarget = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._q_valsTarget_drop = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        self._q_valsActA = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._q_valsActTarget = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._q_valsActA_drop = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        self._q_func = self._q_valsA
        self._q_funcTarget = self._q_valsTarget
        self._q_func_drop = self._q_valsA_drop
        self._q_funcTarget_drop = self._q_valsTarget_drop
        self._q_funcAct = self._q_valsActA
        self._q_funcAct_drop = self._q_valsActA_drop
        
        inputs_ = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getActionSymbolicVariable(): self._q_valsActA,
        }
        self._q_valsA = lasagne.layers.get_output(self._model.getCriticNetwork(), inputs_)
        inputs_ = {
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getActionSymbolicVariable(): self._q_valsActTarget,
        }
        self._q_valsB = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), inputs_)
        
        
        # self._q_funcAct = theano.function(inputs=[State], outputs=self._q_valsActA, allow_input_downcast=True)
        
        self._target = T.mul(T.add(self._model.getRewardSymbolicVariable(), T.mul(self._discount_factor, self._q_valsTargetNextState )), self._Fallen)
        self._diff = self._target - self._q_func
        self._diff_drop = self._target - self._q_func_drop 
        # loss = 0.5 * self._diff ** 2 
        loss = T.pow(self._diff, 2)
        self._loss = T.mean(loss)
        self._loss_drop = T.mean(0.5 * self._diff_drop ** 2)
    
        # assert len(lasagne.layers.helper.get_all_params(self._l_outA)) == 16
        # Need to remove the action layers from these params
        self._params = lasagne.layers.helper.get_all_params(self._model.getCriticNetwork())[-len(lasagne.layers.helper.get_all_params(self._model.getActorNetwork())):] 
        print ("******Number of Layers is: " + str(len(lasagne.layers.helper.get_all_params(self._model.getCriticNetwork()))))
        print ("******Number of Action Layers is: " + str(len(lasagne.layers.helper.get_all_params(self._model.getActorNetwork()))))
        self._actionParams = lasagne.layers.helper.get_all_params(self._model.getActorNetwork())
        self._givens_ = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._Fallen: self._fallen_shared
            # self._model.getActionSymbolicVariable(): self._actions_shared,
        }
        self._actGivens = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            # self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            # self._model.getActionSymbolicVariable(): self._model.getActions(),
            # self._Fallen: self._fallen_shared
            # self._tmp_diff: self._tmp_diff_shared
        }
        
        self._critic_regularization = (self._critic_regularization_weight * 
                                       lasagne.regularization.regularize_network_params(
                                            self._model.getCriticNetwork(), lasagne.regularization.l2))
        
        ## MSE update
        self._value_grad = T.grad(self._loss + self._critic_regularization
                                                     , self._params)
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.rmsprop(self._value_grad
                                                     , self._params, self._learning_rate, self._rho,
                                           self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.momentum(self._value_grad
                                                      , self._params, self._critic_learning_rate , momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.adam(self._value_grad
                        , self._params, self._critic_learning_rate , beta1=0.9, beta2=0.9, epsilon=self._rms_epsilon)
        elif ( self.getSettings()['optimizer'] == 'adagrad'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.adagrad(self._value_grad
                        , self._params, self._critic_learning_rate, epsilon=self._rms_epsilon)
        else:
            print ("Unknown optimization method: ", self.getSettings()['optimizer'])
            sys.exit(-1)
        
        
        
        ### Maximize wrt q function
        
        # theano.gradient.grad_clip(x, lower_bound, upper_bound) # // TODO
        self._actionUpdates = lasagne.updates.adam(T.mean(self._q_func) + 
          (self._decay_weight * lasagne.regularization.regularize_network_params(
              self._model.getActorNetwork(), lasagne.regularization.l2)), self._actionParams, 
                  self._learning_rate, beta1=0.9, beta2=0.9, epsilon=self._rms_epsilon)
        
        
        
        self._train = theano.function([], [self._loss, self._q_func], updates=self._updates_, givens=self._givens_)
        # self._trainActor = theano.function([], [actLoss, self._q_valsActA], updates=actionUpdates, givens=actGivens)
        self._trainActor = theano.function([], [self._q_func], updates=self._actionUpdates, givens=self._actGivens)
        self._q_val = theano.function([], self._q_valsA,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._q_action = theano.function([], self._q_valsActA,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        """
        inputs_ = [
                   self._model.getStateSymbolicVariable(), 
                   self._model.getRewardSymbolicVariable(), 
                   # ResultState
                   ]
        self._bellman_error = theano.function(inputs=inputs_, outputs=self._diff, allow_input_downcast=True)
        """
        # self._diffs = theano.function(input=[State])
        self._bellman_error2 = theano.function(inputs=[], outputs=self._diff, allow_input_downcast=True, givens=self._givens_)
        
    def _trainOneActions(self, states, actions, rewards, result_states):
        print ("Training action")
        # lossActor, _ = self._trainActor()
        State = T.dmatrix("State")
        # State.tag.test_value = np.random.rand(batch_size,self._state_length)
        #ResultState = T.dmatrix("ResultState")
        #ResultState.tag.test_value = np.random.rand(batch_size,self._state_length)
        #Reward = T.col("Reward")
        #Reward.tag.test_value = np.random.rand(batch_size,1)
        Action = T.dmatrix("Action")
        #Action.tag.test_value = np.random.rand(batch_size, self._self._action_length)
        
        
        for state, action, reward, result_state in zip(states, actions, rewards, result_states):
            # print (state)
            # print (action)
            self._states_shared.set_value([state])
            self._next_states_shared.set_value([result_state])
            self._actions_shared.set_value([action])
            self._rewards_shared.set_value([reward])
            # print ("Q value for state and action: " + str(self.q_value([state])))
            # all_paramsA = lasagne.layers.helper.get_all_param_values(self._l_outA)
            # print ("Network length: " + str(len(all_paramsA)))
            # print ("weights: " + str(all_paramsA[0]))
            # lossActor, _ = self._trainActor()
            _params = lasagne.layers.helper.get_all_params(self._l_outA)
            # print (_params[0].get_value())
            inputs_ = {
                State: self._states_shared,
                Action: self._q_valsActA,
            }
            self._q_valsA = lasagne.layers.get_output(self._l_outA, inputs_)
            
            
            updates_ = rmsprop(T.mean(self._q_valsA) + (1e-6 * lasagne.regularization.regularize_network_params(
                self._l_outA, lasagne.regularization.l2)), _params, 
                self._learning_rate * -T.mean(self._diff), self._rho, self._rms_epsilon)
            
            ind = 0
            print ("Update: " + str (updates_.items()))
            print ("Updates length: " + str (len(updates_.items()[ind][0].get_value())) )
            print (" Updates: " + str(updates_.items()[ind][0].get_value()))
            
            
    def updateTargetModel(self):
        # print ("Updating target Model")
        """
            Target model updates
        """
        ## I guess it is okay to lerp the entire network even though we only really want to 
        ## lerp the value function part of the networks, the target policy is not used for anythings
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork())
        all_paramsB = lasagne.layers.helper.get_all_param_values(self._modelTarget.getCriticNetwork())
        lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        
        # print ("l_out length: " + str(len(all_paramsA)))
        # print ("l_out length: " + str(all_paramsA[-6:]))
        # print ("l_out[0] length: " + str(all_paramsA[0]))
        # print ("l_out[4] length: " + str(all_paramsA[4]))
        # print ("l_out[5] length: " + str(all_paramsA[5]))
        # print ("l_out[6] length: " + str(all_paramsA[6]))
        # print ("l_out[7] length: " + str(all_paramsA[7]))
        # print ("l_out[11] length: " + str(all_paramsA[11]))
        # print ("param Values")
        all_params = []
        for paramsA, paramsB in zip(all_paramsA, all_paramsB):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        """
        all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        all_paramsActB = lasagne.layers.helper.get_all_param_values(self._l_outActB)
        # print ("l_outAct[0] length: " + str(all_paramsActA[0]))
        # print ("l_outAct[4] length: " + str(all_paramsActA[4]))
        # print ("l_outAct[5] length: " + str(all_paramsActA[5]))
        all_paramsAct = []
        for paramsA, paramsB in zip(all_paramsActA, all_paramsActB):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_paramsAct.append(params)
            """
        lasagne.layers.helper.set_all_param_values(self._l_outB, all_params)
        # lasagne.layers.helper.set_all_param_values(self._l_outActB, all_paramsAct) 
        
    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget.getCriticNetwork()))
        return params
        
    def setNetworkParameters(self, params):
        lasagne.layers.helper.set_all_param_values(self._model.getCriticNetwork(), params[0])
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getCriticNetwork(), params[1])
        
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
        
    def trainCritic(self, states, actions, rewards, result_states):
        self.setData(states, actions, rewards, result_states, falls)
        
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        loss, _ = self._train()
        return loss
        
    def trainActor(self, states, actions, rewards, result_states):
        self.setData(states, actions, rewards, result_states, falls)
        
        loss = self._trainActor()
        return loss
        
    def train(self, states, actions, rewards, result_states):
        loss = self.trainCritic(states, actions, rewards, result_states)
        lossActor = self.trainActor(states, actions, rewards, result_states)
        return loss
    
