import numpy as np
import sys
sys.path.append('../')
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

class ForwardDynamicsKeras(KERASAlgorithm):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        super(ForwardDynamicsKeras,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        self._model = model
        self._modelTarget = None
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-4
        
        condition_reward_on_result_state = False
        self._train_combined_loss = False
        
        inputs_ = [self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable()] 
        self._model._forward_dynamics_net = Model(inputs=inputs_, outputs=self._model._actor)
        if (print_info):
            print("FD Net summary: ", self._model._forward_dynamics_net.summary())
        self._model._reward_net = Model(inputs=inputs_, outputs=self._model._reward_net)
        if (print_info):
            print("Reward Net summary: ", self._model._reward_net.summary())
        
        self._forward = self._model.getForwardDynamicsNetwork()([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
        self._reward = self._model.getRewardNetwork()([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
        
        ForwardDynamicsKeras.compile(self)
    
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print ("Clipping: ", sgd.decay)
        print("sgd, critic: ", sgd)
        self._model.getRewardNetwork().compile(loss='mse', optimizer=sgd)
        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print("sgd, actor: ", sgd)
        print ("Clipping: ", sgd.decay)
        self._model.getForwardDynamicsNetwork().compile(loss='mse', optimizer=sgd)

        self._params = self._model.getForwardDynamicsNetwork().trainable_weights        
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
        
        self.fd = K.function([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable(), K.learning_phase()], [self._forward])
        self.reward = K.function([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable(), K.learning_phase()], [self._reward])
        
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model.getForwardDynamicsNetwork().get_weights()))
        params.append(copy.deepcopy(self._model.getRewardNetwork().get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        self._model.getForwardDynamicsNetwork().set_weights(params[0])
        self._model.getRewardNetwork().set_weights( params[1] )
        
    def setData(self, states, actions, result_states=None, rewards=None):
        pass
        """
        self._model.setStates(states)
        if not (result_states is None):
            self._model.setResultStates(result_states)
        self._model.setActions(actions)
        if not (rewards is None):
            self._model.setRewards(rewards)
            """
            
    def setGradTarget(self, grad):
        self._fd_grad_target_shared.set_value(grad)
        
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
    
    def getRewardGrads(self, states, actions, alreadyNormed=False):
        # states = np.array(states, dtype=self.getSettings()['float_type'])
        # actions = np.array(actions, dtype=self.getSettings()['float_type'])
        if ( alreadyNormed is False ):
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            # rewards = np.array(norm_state(rewards, self._reward_bounds), dtype=self.getSettings()['float_type'])
        # self.setData(states, actions)
        return self._get_grad_reward([states, actions, 0])[0]
                
    def train(self, states, actions, result_states, rewards, updates=1, batch_size=None, lstm=False, datas=[], trainInfo=None):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        # self.setData(states, actions, result_states, rewards)
        # print ("Performing Critic trainning update")
        #if (( self._updates % self._weight_update_steps) == 0):
        #    self.updateTargetModel()
        self._updates += 1
        if (batch_size is None):
            batch_size_=states.shape[0]
        else:
            batch_size_=batch_size
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        if ( self._train_combined_loss ):
            pass
            # loss = self._train_combined()
            # loss = self._train_combined()
        else:
            score = self._model.getForwardDynamicsNetwork().fit([states, actions], result_states,
              epochs=updates, batch_size=batch_size_,
              verbose=0,
              shuffle=True
              # callbacks=[early_stopping],
              )
            loss = score.history['loss'][0]
            if ( self.getSettings()['train_reward_predictor']):
                # print ("self._reward_bounds: ", self._reward_bounds)
                # print( "Rewards, predicted_reward, difference, model diff, model rewards: ", np.concatenate((rewards, self._predict_reward(), self._predict_reward() - rewards, self._reward_error(), self._reward_values()), axis=1))
                score = self._model.getRewardNetwork().fit([states, actions], rewards,
                  epochs=updates, batch_size=batch_size_,
                  verbose=0,
                  shuffle=True
                  # callbacks=[early_stopping],
                  )
                lossReward = score.history['loss'][0]
                if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                    print ("Loss Reward: ", lossReward)
            if ( 'train_state_encoding' in self.getSettings() and (self.getSettings()['train_state_encoding'])):
                pass
                # lossEncoding = self._train_state_encoding()
                # if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                #    print ("Loss Encoding: ", lossEncoding)     
        # This undoes the Actor parameter updates as a result of the Critic update.
        # print (diff_)
        return loss
    
    def predict(self, state, action):
        # print("State: ", state)
        # print("Action: ", action)
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        pred = self.fd([state, action,0])[0]
        pred = pred[:,:self._state_length] ### Cut off std
        # print ("pred: ", pred[:,:self._action_length])
        state_ = scale_state(pred, self._state_bounds)
        return state_
    
    def predictWithDropout(self, state, action):
        # print("State: ", state)
        # print("Action: ", action)
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        pred = self.fd([state, action,1])[0]
        pred = pred[:,:self._state_length] ### Cut off std
        state_ = scale_state(pred, self._state_bounds)
        return state_
    
    def predict_std(self, state, action, p=1.0):
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        state_ = self._forwardDynamics_std() * (action_bound_std(self._state_bounds))
        return state_
    
    def predict_reward(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        predicted_reward = self.reward([state, action, 0])[0]
        reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_reward(predicted_reward, self.getRewardBounds())[0] * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_state(predicted_reward, self._reward_bounds)
        # print ("reward, predicted reward: ", reward_, predicted_reward)
        return reward_
    
    def predict_batch(self, states, actions):
        ## These input should already be normalized.
        return self.fd([states, actions, 0])[0]
    
    def predict_reward_batch(self, states, actions):
        """
            This data should already be normalized
        """
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        predicted_reward = self.reward([states, actions, 0])[0]
        return predicted_reward

    def bellman_error(self, states, actions, result_states, rewards):
        self.setData(states, actions, result_states, rewards)
        predicted_y = self.predict(states, actions)
        diff = np.mean(np.abs(predicted_y - result_states))
        return diff
    
    def reward_error(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        predicted_y = self.predict_reward(states, actions)
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
        self._model._forward_dynamics_net.save(fileName+"_FD"+suffix, overwrite=True)
        self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
        try:
            from keras.utils import plot_model
            ### Save model design as image
            plot_model(self._model._forward_dynamics_net, to_file=fileName+"_fd"+'.svg', show_shapes=True)
            plot_model(self._model._reward_net, to_file=fileName+"_reward"+'.svg', show_shapes=True)
        except Exception as inst:
            ### Maybe the needed libraries are not available
            print ("Error saving diagrams for rl models.")
            print (inst)
        # print ("self._model._actor_train: ", self._model._actor_train)
        
    def loadFrom(self, fileName):
        from keras.models import load_model
        import h5py
        suffix = ".h5"
        print ("Loading agent: ", fileName)
        # with K.get_session().graph.as_default() as g:
        self._model._forward_dynamics_net = load_model(fileName+"_FD"+suffix)
        self._model._reward_net = load_model(fileName+"_reward"+suffix)
        if (self._modelTarget is not None):
            self._modelTarget._forward_dynamics_net = load_model(fileName+"_FD"+suffix)
            self._modelTarget._reward_net = load_model(fileName+"_reward"+suffix)
        # self._model._actor_train = load_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        # self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        hf.close()