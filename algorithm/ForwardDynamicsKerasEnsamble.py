import numpy as np
import sys
from model.ModelUtil import *
from model.LearningUtil import loglikelihood, loglikelihoodMEAN, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat, likelihood, loglikelihoodMEAN
from keras.optimizers import SGD
from model.LearningUtil import loglikelihood_keras, likelihood_keras, kl_keras, kl_D_keras, entropy_keras
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras
from keras.models import Sequential, Model

# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.KERASAlgorithm import KERASAlgorithm

class ForwardDynamicsKerasEnsamble(KERASAlgorithm):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        super(ForwardDynamicsKerasEnsamble,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        self._fd_ensemble_size = self.getSettings()['fd_ensemble_size']
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-6
        
        model_ = model
        self._modelTarget = None
        inputs_ = [model_.getStateSymbolicVariable(), model_.getActionSymbolicVariable()] 
        model_._forward_dynamics_net = Model(inputs=inputs_, outputs=model_._actor)
        if (print_info):
            print("FD Net summary: ", model_._forward_dynamics_net.summary())
        model_._reward_net = Model(inputs=inputs_, outputs=model_._reward_net)
        if (print_info):
            print("Reward Net summary: ", model_._reward_net.summary())
        
        condition_reward_on_result_state = False
        self._train_combined_loss = False
        
        self._models = [model_]
        
        def loss_std(action_true, action_pred):
            action_true = action_true[:,:self._state_length]
            action_pred_mean = action_pred[:,:self._state_length]
            action_pred_std = action_pred[:,self._state_length:]
            prob = loglikelihood_keras(action_true, action_pred_mean, action_pred_std, self._state_length)
            # entropy = 0.5 * T.mean(T.log(2 * np.pi * action_pred_std + 1 ) )
            # actLoss = -1.0 * (K.mean(K.mean(prob, axis=-1)) + (entropy * 1e-2))
            actLoss = -1.0 * (K.mean(K.mean(prob, axis=-1)))
            ### Average over batch
            return actLoss
        
        self._loss_std = loss_std
        
        for i in range(1, self._fd_ensemble_size):
            ### create new instance of model
            model_ = type(model)(state_length, action_length, state_bounds, 
                                 action_bounds, reward_bounds, settings_, print_info=print_info)
            inputs_ = [model_.getStateSymbolicVariable(), model_.getActionSymbolicVariable()]
            ### Compile networks 
            model_._forward_dynamics_net = Model(inputs=inputs_, outputs=model_._actor)
            if (print_info):
                print("FD Net summary: ", model_._forward_dynamics_net.summary())
            model_._reward_net = Model(inputs=inputs_, outputs=model_._reward_net)
            if (print_info):
                print("Reward Net summary: ", model_._reward_net.summary())
                
            self._models.append(model_)
            
        for i in range(self._fd_ensemble_size):            
            sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
            print ("Clipping: ", sgd.decay)
            print("sgd, critic: ", sgd)
            self._models[i].getRewardNetwork().compile(loss='mse', optimizer=sgd)
            # sgd = SGD(lr=0.0005, momentum=0.9)
            sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
            if ("use_stochastic_forward_dynamics" in self.getSettings()
                and (self.getSettings()['use_stochastic_forward_dynamics'] == True)):
                self._models[i].getForwardDynamicsNetwork().compile(loss=loss_std, optimizer=sgd)
            else:
                self._models[i].getForwardDynamicsNetwork().compile(loss='mse', optimizer=sgd)
        
        ### data types for model
        self._fd_grad_target = T.matrix("FD_Grad")
        self._fd_grad_target.tag.test_value = np.zeros((self._batch_size,self._state_length), dtype=np.dtype(self.getSettings()['float_type']))
        self._fd_grad_target_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                      dtype=self.getSettings()['float_type']))
        
        ##
        
        
        ForwardDynamicsKerasEnsamble.compile(self)
    
    def compile(self):

        self._forwards = []
        self._rewards = []        
        self.fds = []
        self.rewards = []
        for i in range(self._fd_ensemble_size):
            self._forwards.append(self._models[i].getForwardDynamicsNetwork()([self._models[i].getStateSymbolicVariable(), self._models[i].getActionSymbolicVariable()]))
            self._rewards.append(self._models[i].getRewardNetwork()([self._models[i].getStateSymbolicVariable(), self._models[i].getActionSymbolicVariable()]))
            

        # sgd = SGD(lr=0.001, momentum=0.9)
        for i in range(self._fd_ensemble_size):
            
            self.fds.append(K.function([self._models[i].getStateSymbolicVariable(), self._models[i].getActionSymbolicVariable(), K.learning_phase()], [self._forwards[i]]))
            self.rewards.append(K.function([self._models[i].getStateSymbolicVariable(), self._models[i].getActionSymbolicVariable(), K.learning_phase()], [self._rewards[i]]))
        
    def getNetworkParameters(self):
        params = []
        for i in range(self._fd_ensemble_size):
            params.append(copy.deepcopy(self._models[i].getForwardDynamicsNetwork().get_weights()))
            params.append(copy.deepcopy(self._models[i].getRewardNetwork().get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        for i in range(self._fd_ensemble_size):
            self._model.getForwardDynamicsNetwork().set_weights(params[(i*2) + 0])
            self._model.getRewardNetwork().set_weights( params[(i*2) + 1] )
        
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
        # print ("grad: ", grad)
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
                
    def train(self, states, actions, result_states, rewards, updates=1, batch_size=None, lstm=False):
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
            losses = []
            member = np.random.randint(low=0, high=self._fd_ensemble_size)
            # for i in range(self._fd_ensemble_size):
            score = self._models[member].getForwardDynamicsNetwork().fit([states, actions], result_states,
              epochs=updates, batch_size=batch_size_,
              verbose=0,
              shuffle=True
              # callbacks=[early_stopping],
              )
            losses.append(score.history['loss'][0])
            if ( self.getSettings()['train_reward_predictor']):
                # print ("self._reward_bounds: ", self._reward_bounds)
                # print( "Rewards, predicted_reward, difference, model diff, model rewards: ", np.concatenate((rewards, self._predict_reward(), self._predict_reward() - rewards, self._reward_error(), self._reward_values()), axis=1))
                score = self._models[member].getRewardNetwork().fit([states, actions], rewards,
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
        return np.mean(losses)
    
    def predict(self, state, action, member=0):
        # print("State: ", state)
        # print("Action: ", action)
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        state_ = scale_state(np.array(self._models[member].getForwardDynamicsNetwork().predict([state, action]))[:,:self._state_length], self._state_bounds)
        # print("state:", state_)
        return state_
    
    def predictWithDropout(self, state, action, member=0):
        # print("State: ", state)
        # print("Action: ", action)
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        # state_ = scale_state(self.fds[member]([state, action,1])[0][:][:self._state_length], self._state_bounds)
        state_ = scale_state(np.array(self.fds[member]([state, action,1]))[0,:,:self._state_length], self._state_bounds)
        # print("state dropout:", state_)
        return state_
    
    def predict_std(self, state, action, p=1.0, member=0):
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        if ("use_stochastic_forward_dynamics" in self.getSettings()
            and (self.getSettings()['use_stochastic_forward_dynamics'] == True)):
            state_ = np.array(self.fds[member]([state, action,0]))[0,:,self._state_length:] * (action_bound_std(self._state_bounds))
        else:
            state_ = self.predict(state, action, member) * 0
        # state_ = self._forwardDynamics_std() * (action_bound_std(self._state_bounds))
        return state_
    
    def predict_reward(self, state, action, member=0):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        predicted_reward = self.rewards[member]([state, action, 0])[0]
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
        self._models[0]._forward_dynamics_net.save(fileName+"_FD"+suffix, overwrite=True)
        self._models[0]._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
        # print ("self._model._actor_train: ", self._model._actor_train)
        
    def loadFrom(self, fileName):
        from keras.models import load_model
        import h5py
        suffix = ".h5"
        print ("Loading agent: ", fileName)
        self._models[0]._forward_dynamics_net = load_model(fileName+"_FD"+suffix, custom_objects={'loss_std': self._loss_std})
        self._models[0]._reward_net = load_model(fileName+"_reward"+suffix)
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        hf.close()
        
        self.compile()
        