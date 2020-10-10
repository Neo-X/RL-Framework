import numpy as np
import sys
from model.ModelUtil import *
from model.LearningUtil import loglikelihood, loglikelihoodMEAN, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat, likelihood, loglikelihoodMEAN
from model.LearningUtil import loglikelihood, likelihood, likelihoodMEAN, kl, kl_D, entropy, flatgrad, zipsame, get_params_flat, setFromFlat
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras
from keras.models import Sequential, Model

# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.KERASAlgorithm import KERASAlgorithm

class StateTransformationKeras(KERASAlgorithm):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        super(StateTransformationKeras,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        self._model = model
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-6
        self._modelTarget = None
        
        condition_reward_on_result_state = False
        self._train_combined_loss = False
        
        inputs_ = [self._model.getStateSymbolicVariable()] 
        if ("use_single_network" in self._settings and
               (self._settings['use_single_network'] == True)):
            self._model.setTransformationDynamicsNetwork(Model(inputs=inputs_, outputs=self._model._trans))
        else:
            self._model.setTransformationDynamicsNetwork(Model(inputs=inputs_, outputs=self._model._actor))
        if (print_info):
            print("Transformation Net summary: ", self._model.getTransformationDynamicsNetwork().summary())
        
        
        ##
        
        self._forward = self._model.getTransformationDynamicsNetwork()([self._model.getStateSymbolicVariable()])
        
        StateTransformationKeras.compile(self)
    
    def compile(self):

        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.90), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print("sgd, actor: ", sgd)
        print ("Clipping: ", sgd.decay)
        self._model.getTransformationDynamicsNetwork().compile(loss='mse', optimizer=sgd)

        self._params = self._model.getTransformationDynamicsNetwork().trainable_weights        
        """
        weights = [self._model.getActionSymbolicVariable()]
        gradients = K.gradients(T.mean(self._q_function), [self._model.getStateSymbolicVariable()]) # gradient tensors
        ### DPG related functions
        self._get_gradients = K.function(inputs=[self._model.getStateSymbolicVariable()], outputs=gradients)
        """
        ### Get reward input grad
        weights = [self._model.getActionSymbolicVariable()]
        # reward_gradients = K.gradients(T.mean(self._reward), [self._model.getActionSymbolicVariable()]) # gradient tensors
        ### DPG related functions
        #self._get_grad_reward = K.function(inputs=[self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable(), K.learning_phase()], outputs=reward_gradients)
        
        
        # self._get_grad = theano.function([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable(), K.learning_phase()], outputs=T.grad(cost=None, wrt=[self._model.getActionSymbolicVariable()] + self._params,
        #                                                    known_grads={self._forward: self._fd_grad_target_shared}), 
        #                                 allow_input_downcast=True)
        
        # self._get_grad_reward = theano.function([], outputs=lasagne.updates.get_or_compute_grads((self._reward_loss_NoDrop), [lasagne.layers.get_all_layers(self._model.getRewardNetwork())[0].input_var] + self._reward_params), allow_input_downcast=True,
        # self._get_grad_reward = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._reward), [self._model.getActionSymbolicVariable()Var] + self._reward_params), allow_input_downcast=True, 
        #                                         givens=self._inputs_reward_)
        
        self.fd = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self._forward])
        
        
        self._fd_regularization = K.sum(self._model.getTransformationDynamicsNetwork().losses)
        self._get_fd_regularization = K.function([], [self._fd_regularization])
        
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model.getTransformationDynamicsNetwork().get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        self._model.getTransformationDynamicsNetwork().set_weights(params[0])
            
    def getGrads(self, states, actions, result_states, v_grad=None, alreadyNormed=False):
        if ( alreadyNormed == False ):
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            result_states = np.array(norm_state(result_states, self._state_bounds), dtype=self.getSettings()['float_type'])
        # result_states = np.array(result_states, dtype=self.getSettings()['float_type'])
        # self.setData(states, actions, result_states)
        # if (v_grad != None):
        # print ("states shape: ", states.shape, " actions shape: ", actions.shape, " v_grad.shape: ", v_grad.shape)
        self.setGradTarget(v_grad)
        # print ("states shape: ", states.shape, " actions shape: ", actions.shape)
        # grad = self._get_grad([states, actions])[0]
        grad = np.zeros_like(states)
        print ("grad: ", grad)
        return grad
    
    def train(self, states, actions, result_states, rewards, updates=1, 
              batch_size=None, lstm=None, datas=None, trainInfo=None):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        # self.setData(states, actions, result_states, rewards)
        # print ("Performing Critic trainning update")
        print ("states: ", states.shape)
        print ("result_states: ", result_states.shape)
        #if (( self._updates % self._weight_update_steps) == 0):
        #    self.updateTargetModel()
        self._updates += 1
        if (batch_size is None):
            batch_size_=states.shape[0]
        else:
            batch_size_=batch_size
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        score = self._model.getTransformationDynamicsNetwork().fit([states], result_states,
          epochs=updates, batch_size=batch_size_,
          verbose=0,
          shuffle=True
          # callbacks=[early_stopping],
          )
#         loss = np.mean(score.history['loss'])
        # This undoes the Actor parameter updates as a result of the Critic update.
        # print (diff_)
        return score.history
    
    def predict(self, state, action):
        # print("State: ", state)
        # print("Action: ", action)
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # state_ = scale_state(self.fd([state, 0])[0], self.getResultStateBounds())
        state_ = self.fd([state, 0])[0]
        return state_
    
    def predictWithDropout(self, state, action):
        # print("State: ", state)
        # print("Action: ", action)
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        state_ = scale_state(self.fd([state, 1])[0], self._state_bounds)
        return state_
    
    def predict_std(self, state, action, p=1.0):
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        state_ = self._forwardDynamics_std() * (action_bound_std(self._state_bounds))
        return state_
    
    def predict_batch(self, states, actions):
        ## These input should already be normalized.
        return self.fd([states, actions, 0])[0]

    def bellman_error(self, states, actions, result_states, rewards):
        predicted_y = self.predict(states, actions)
        diff = np.mean(np.abs(predicted_y - result_states))
        return diff
    
    def saveTo(self, fileName):
        # print(self, "saving model")
        import h5py
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
        self._model.getTransformationDynamicsNetwork().save(fileName+"_FD"+suffix, overwrite=True)
        # print ("self._model._actor_train: ", self._model._actor_train)
        
    def loadFrom(self, fileName):
        from keras.models import load_model
        import h5py
        suffix = ".h5"
        print ("Loading agent: ", fileName)
        # with K.get_session().graph.as_default() as g:
        self._model.setTransformationDynamicsNetwork( load_model(fileName+"_FD"+suffix))
        if (self._modelTarget is not None):
            self._modelTarget._forward_dynamics_net = load_model(fileName+"_FD"+suffix)
        # self._model._actor_train = load_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        # self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        hf.close()