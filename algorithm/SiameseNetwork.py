import theano
from theano import tensor as T
import numpy as np
# import lasagne
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

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    ####           Make these smaller               While making these bigger
    return K.mean((y_true * K.square(y_pred)) + ((1 - y_true) * K.square(K.maximum(margin - y_pred, 0))))

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

def create_pairs2(x):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    noise_scale = 0.02
    pair1 = []
    pair2 = []
    labels = []
    n = x.shape[0] - 1
    indices = list(np.random.randint(low=0, high=n, size=n))
    for i in range(n):
        ### Identical pair
        i = indices[i]
        noise = np.random.normal(loc=0, scale=noise_scale, size=x[i].shape)
        x1 = [x[i] + noise]
        noise = np.random.normal(loc=0, scale=noise_scale, size=x[i].shape)
        x2 = [x[i] + noise]
        if (np.random.rand() > 0.5):
            pair1 += x1
            pair2 += x2
        else:
            pair1 += x2
            pair2 += x1
        ### Different pair
        z=i
        while (z == i): ## get result that is not the same
            z = np.random.randint(low=0, high=n)
        noise = np.random.normal(loc=0, scale=noise_scale, size=x[i].shape)
        x1 = [x[i] + noise]
        noise = np.random.normal(loc=0, scale=noise_scale, size=x[i].shape)
        x2 = [x[z] + noise]
        if (np.random.rand() > 0.5):
            pair1 += x1
            pair2 += x2
        else:
            pair1 += x2
            pair2 += x1
        labels += [[1], [0]]
    return np.array(pair1), np.array(pair2), np.array(labels)

class SiameseNetwork(KERASAlgorithm):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        super(SiameseNetwork,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        self._model = model
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-6
        
        condition_reward_on_result_state = False
        self._train_combined_loss = False

        inputs_ = [self._model.getStateSymbolicVariable()] 
        self._model._forward_dynamics_net = Model(inputs=inputs_, outputs=self._model._forward_dynamics_net)
        if (print_info):
            print("FD Net summary: ", self._model._forward_dynamics_net.summary())
        
        inputs_ = [self._model.getStateSymbolicVariable()] 
        self._model._reward_net = Model(inputs=inputs_, outputs=self._model._reward_net)
        if (print_info):
            print("FD Reward Net summary: ", self._model._reward_net.summary())

        self._modelTarget = None
        SiameseNetwork.compile(self)
    
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        
        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = self._model._forward_dynamics_net(self._model.getStateSymbolicVariable())
        processed_b = self._model._forward_dynamics_net(self._model.getResultStateSymbolicVariable())
        
        distance = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        self._model._forward_dynamics_net = Model(inputs=[self._model.getStateSymbolicVariable(),
                               self._model.getResultStateSymbolicVariable()], outputs=distance)

        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print("sgd, actor: ", sgd)
        print ("Clipping: ", sgd.decay)
        self._model._forward_dynamics_net.compile(loss=contrastive_loss, optimizer=sgd)

        
        self._contrastive_loss = K.function([self._model.getStateSymbolicVariable(), 
                                             self._model.getResultStateSymbolicVariable(),
                                             K.learning_phase()], 
                                            [distance])
        # self.reward = K.function([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable(), K.learning_phase()], [self._reward])
        
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model._forward_dynamics_net.get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        self._model._forward_dynamics_net.set_weights(params[0])
        
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
                
    def train(self, states, actions, result_states, rewards, updates=1, batch_size=None):
        """
            states will come for the agent and
            results_states can come from the imitation agent
        """
        if ("replace_next_state_with_imitation_viz_state" in self.getSettings()
            and (self.getSettings()["replace_next_state_with_imitation_viz_state"] == True)):
            states = np.concatenate((states, result_states), axis=0)
        te_pair1, te_pair2, te_y = create_pairs2(states)
        self._updates += 1
        if (batch_size is None):
            batch_size_=states.shape[0]
        else:
            batch_size_=batch_size
            
        loss = 0
        dist_ = np.array(self._contrastive_loss([te_pair1, te_pair2, 0]))[0]
        dist = np.mean(dist_)
        te_y = np.array(te_y)
        # print("Distance: ", dist)
        # print("targets: ", te_y)
        # print("pairs: ", te_pair1)
        # print("Distance.shape, targets.shape: ", dist_.shape, te_y.shape)
        # print("Distance, targets: ", np.concatenate((dist_, te_y), axis=1))
        if ( dist > 0):
            score = self._model._forward_dynamics_net.fit([te_pair1, te_pair2], te_y,
              epochs=updates, batch_size=batch_size_,
              verbose=0,
              shuffle=True
              )
            loss = np.mean(score.history['loss'])
            # print ("loss: ", loss)
        return loss
    
    def predict(self, state, state2):
        """
            Compute distance between two states
        """
        # print("state shape: ", np.array(state).shape)
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        state2 = np.array(norm_state(state2, self._state_bounds), dtype=self.getSettings()['float_type'])
        state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
        # dist_ = np.array(self._contrastive_loss([te_pair1, te_pair2, 0]))[0]
        # print("state_ shape: ", np.array(state_).shape)
        return state_
    
    def predictWithDropout(self, state, action):
        # "dropout"
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        state_ = scale_state(self._forwardDynamics_drop()[0], self._state_bounds)
        return state_
    
    def predict_std(self, state, action, p=1.0):
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        state_ = self._forwardDynamics_std() * (action_bound_std(self._state_bounds))
        return state_
    
    def predict_reward(self, state, action):
        """
            Predict reward which is inverse of distance metric
        """
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        predicted_reward = self.reward([state, action, 0])[0]
        reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
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
        
        states = np.concatenate((states, result_states), axis=0)
        te_pair1, te_pair2, te_y = create_pairs2(states)
        
        predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
        te_acc = compute_accuracy(predicted_y, te_y)
        return te_acc
    
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
        # self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
        # print ("self._model._actor_train: ", self._model._actor_train)
        
    def loadFrom(self, fileName):
        import h5py
        from keras.models import load_model
        suffix = ".h5"
        print ("Loading agent: ", fileName)
        # with K.get_session().graph.as_default() as g:
        self._model._forward_dynamics_net = load_model(fileName+"_FD"+suffix, custom_objects={'contrastive_loss': contrastive_loss})
        self._forward_dynamics_net = self._model._forward_dynamics_net
        print ("******** self._forward_dynamics_net: ", self._forward_dynamics_net)
        # self._model._reward_net = load_model(fileName+"_critic"+suffix)
        if (self._modelTarget is not None):
            self._modelTarget._forward_dynamics_net = load_model(fileName+"_actor_T"+suffix)
            self._modelTarget._reward_net = load_model(fileName+"_reward_net_T"+suffix)
        # self._model._actor_train = load_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        # self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        hf.close()
        