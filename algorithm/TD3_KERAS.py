import numpy as np
import sys
import copy
from model.ModelUtil import *
from algorithm.KERASAlgorithm import KERASAlgorithm
from model.LearningUtil import loglikelihood, kl, entropy, change_penalty
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras
from keras.models import Sequential, Model
from keras.layers import RepeatVector


# For debugging
# theano.config.mode='FAST_COMPILE'
from collections import OrderedDict
def neg_y(true_y, pred_y):
    return K.mean(-pred_y)

class TD3_KERAS(KERASAlgorithm):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False):
        """
            In order to get this to work we need to be careful not to update the actor parameters
            when updating the critic. This can be an issue when the Concatenating networks together.
            The first first network becomes a part of the second. However you can still access the first
            network by itself but an updates on the second network will effect the first network.
            Care needs to be taken to make sure only the parameters of the second network are updated.
        """
        
        super(TD3_KERAS, self).__init__( model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False)
        
        self._c = 0.1
        self._noise_scale = 0.05

        self._model._actor = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=self._model._actor)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Actor summary: ", self._model._actor.summary())
        
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)
             and False):
            self._model._critic = Model(inputs=[self._model.getResultStateSymbolicVariable(),
                                              self._model.getActionSymbolicVariable()], outputs=self._model._critic)
        else:
            self._model._critic = Model(inputs=[self._model.getStateSymbolicVariable(),
                                              self._model.getActionSymbolicVariable()], outputs=self._model._critic)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Critic summary: ", self._model._critic.summary())
            
        self._model1 = type(self._model)(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False)
        self._model1._actor = Model(inputs=[self._model1.getStateSymbolicVariable()], outputs=self._model1._actor)
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)
             and False):
            self._model1._critic = Model(inputs=[self._model1.getResultStateSymbolicVariable(),
                                              self._model1.getActionSymbolicVariable()], outputs=self._model1._critic)
        else:
            self._model1._critic = Model(inputs=[self._model1.getStateSymbolicVariable(),
                                              self._model1.getActionSymbolicVariable()], outputs=self._model1._critic)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Critic1 summary: ", self._model1._critic.summary())
        
        self._modelTarget = type(self._model)(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False)
        self._modelTarget._actor = Model(inputs=[self._modelTarget.getStateSymbolicVariable()], outputs=self._modelTarget._actor)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Target Actor summary: ", self._modelTarget._actor.summary())
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)
             and False):
            self._modelTarget._critic = Model(inputs=[self._modelTarget.getResultStateSymbolicVariable(),
                                                  self._modelTarget.getActionSymbolicVariable()], outputs=self._modelTarget._critic)
        else:
            self._modelTarget._critic = Model(inputs=[self._modelTarget.getStateSymbolicVariable(),
                                                  self._modelTarget.getActionSymbolicVariable()], outputs=self._modelTarget._critic)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Target Critic summary: ", self._modelTarget._critic.summary())
            
        self._modelTarget1 = type(self._model)(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False)
        self._modelTarget1._actor = Model(inputs=[self._modelTarget1.getStateSymbolicVariable()], outputs=self._modelTarget1._actor)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Target Actor summary: ", self._modelTarget1._actor.summary())
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)
             and False):
            self._modelTarget1._critic = Model(inputs=[self._modelTarget1.getResultStateSymbolicVariable(),
                                                  self._modelTarget1.getActionSymbolicVariable()], outputs=self._modelTarget1._critic)
        else:
            self._modelTarget1._critic = Model(inputs=[self._modelTarget1.getStateSymbolicVariable(),
                                                  self._modelTarget1.getActionSymbolicVariable()], outputs=self._modelTarget1._critic)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("Target Critic summary: ", self._modelTarget1._critic.summary())
        
        
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']

        q_loss_name = 'mse'
        if "q_loss_name" in self.getSettings():
            q_loss_name = self.getSettings()["q_loss_name"]

        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0,
                                    clipvalue=1.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Clipping: ", sgd.decay)
        self._model.getCriticNetwork().compile(loss=q_loss_name, optimizer=sgd)
        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0,
                                    clipvalue=1.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Clipping: ", sgd.decay)
        self._model1.getCriticNetwork().compile(loss=q_loss_name, optimizer=sgd)
        
        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0,
                                    clipvalue=1.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Clipping: ", sgd.decay)
        self._modelTarget.getCriticNetwork().compile(loss=q_loss_name, optimizer=sgd)
        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0,
                                    clipvalue=1.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Clipping: ", sgd.decay)
        self._modelTarget1.getCriticNetwork().compile(loss=q_loss_name, optimizer=sgd)
        
        TD3_KERAS.compile(self)
        
    def compile(self):
        
        self._q_valsActA = self._model.getActorNetwork()([self._model.getStateSymbolicVariable()])[:,:self._action_length]
        self._q_valsActTarget_State = self._modelTarget.getActorNetwork()([self._model.getStateSymbolicVariable()])[:,:self._action_length]
        # self._q_valsActTarget_ResultState = self._modelTarget.getActorNetwork()([self._model.getResultStateSymbolicVariable()])[:,:self._action_length]

        self._q_valsActASTD = ( K.ones_like(self._q_valsActA)) * self.getSettings()['exploration_rate']
        self._q_valsActTargetSTD = (K.ones_like(self._q_valsActTarget_State)) * self.getSettings()['exploration_rate']
                
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)
             and False):
            self._q_function = self._model.getCriticNetwork()([self._model.getResultStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            self._q_function1 = self._model1.getCriticNetwork()([self._model.getResultStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            self._q_function_Target = self._model.getCriticNetwork()([self._model.getResultStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            self._q_function1_Target = self._model1.getCriticNetwork()([self._model.getResultStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            # self._q_func = K.function([self._model.getResultStateSymbolicVariable(), self._model.getActionSymbolicVariable()], [self._q_function])
        else:
            self._q_function = self._model.getCriticNetwork()([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            self._q_function1 = self._model1.getCriticNetwork()([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            self._q_function_Target = self._model.getCriticNetwork()([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            self._q_function1_Target = self._model1.getCriticNetwork()([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable()])
            # self._q_func = K.function([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable()], [self._q_function])
        
        q_vals_b = self._q_function_Target
        target_tmp_ = self._model.getRewardSymbolicVariable() + ((self._discount_factor * q_vals_b ))
        diff = target_tmp_ - self._q_function
        self._q_loss = K.mean(K.mean(diff, axis=-1))
        # print ("Initial W " + str(self._w_o.get_value()) )
        
        # self._get_gradients = K.function(inputs=[self._model._stateInput], outputs=gradients)
        
        
        self._q_action_std = K.function([self._model._stateInput], [self._q_valsActASTD])
        
        # For the combined model we will only train the actor
        self._model.getCriticNetwork().trainable = False
        self._model1.getCriticNetwork().trainable = False
        
        self._act = self._model.getActorNetwork()(
                                [self._model.getStateSymbolicVariable()])
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)
             and False):
            self._qFunc = (self._model.getCriticNetwork()(
                            [self._model.getResultStateSymbolicVariable(), 
                             self._act]))
            self._combined = Model(input=[self._model.getStateSymbolicVariable(),
                                          self._model.getResultStateSymbolicVariable()], 
                                output=self._qFunc)
        else:
            if ("policy_connections" in self.getSettings()
                and (any([self.getSettings()["agent_id"] == m[1] for m in self.getSettings()["policy_connections"]])) ):
                pass
            else:
                self._qFunc = (self._model.getCriticNetwork()(
                                [self._model.getStateSymbolicVariable(), 
                                 self._act]))
                self._combined = Model(input=[self._model.getStateSymbolicVariable()], 
                                    output=self._qFunc)
        if ("policy_connections" in self.getSettings()
            and (any([self.getSettings()["agent_id"] == m[1] for m in self.getSettings()["policy_connections"]])) ):
                pass
        else:
            sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['learning_rate']), 
                                        beta_1=np.float32(0.9), beta_2=np.float32(0.999), 
                                        epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0000001),
                                        amsgrad=False)
            self._combined.compile(loss=[neg_y], optimizer=sgd)
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print("combined qFun Net summary: ",  self._combined.summary())
            
        if (self.getSettings()["regularization_weight"] > 0.0000001):
            self._actor_regularization = K.sum(self._model.getActorNetwork().losses)
        else:
            self._actor_regularization = K.sum(self._model.getActorNetwork().losses)
        
        if (self.getSettings()["critic_regularization_weight"] > 0.0000001):
            self._critic_regularization = K.sum(self._model.getCriticNetwork().losses)
        else:
            self._critic_regularization = K.sum(self._model.getCriticNetwork().losses)
            
        self._get_actor_regularization = K.function([], [self._actor_regularization])
        self._get_critic_regularization = K.function([], [self._critic_regularization])
        self._get_critic_loss = K.function([self._model.getStateSymbolicVariable(),
                                            self._model.getActionSymbolicVariable(),
                                            self._model.getRewardSymbolicVariable(), 
                                            self._model.getResultStateSymbolicVariable(),
                                            K.learning_phase()
                                            ], [self._q_loss])
        if ("policy_connections" in self.getSettings()
            and (any([self.getSettings()["agent_id"] == m[1] for m in self.getSettings()["policy_connections"]])) ):
                pass
        else:
            self._get_actor_loss = K.function([self._model.getStateSymbolicVariable()
                                                 # ,K.learning_phase()
                                                 ], [self._qFunc])
        
    def setFrontPolicy(self, lowerPolicy):
        
        from model.DeepNNKerasAdaptive import keras_slice_3d
        
        
        ### For the combined model we will only train the actor
        self._model.getCriticNetwork().trainable = False
        self._model1.getCriticNetwork().trainable = False
        ### I hope this won't cause the llp to not train...
        lowerPolicy.trainable = False
        
        self._LLP_State = keras.layers.Input(shape=(self.getSettings()["hlc_timestep"]* keras.backend.int_shape(lowerPolicy._model.getStateSymbolicVariable())[1],), name="llp_state")
        llp_state = keras.layers.Reshape((self.getSettings()["hlc_timestep"], keras.backend.int_shape(lowerPolicy._model.getStateSymbolicVariable())[1]))(self._LLP_State)
        print ("self._LLP_State: ", self._LLP_State)
        ### Should the target policy be used here?
        self._llp = lowerPolicy._model._actor
        self._llp_T = lowerPolicy._modelTarget._actor
        self._llp.trainable = False
        self._llp_T.trainable = False
        g = self._model._actor([self._model.getStateSymbolicVariable()])
        ### a <- pi(a|s,g)
        s_llp = keras.layers.core.Lambda(keras_slice_3d, output_shape=(self.getSettings()["hlc_timestep"],4),
                        arguments={'begin': 0, 
                        'end': 4})(llp_state)
                         
        # s_llp = keras.layers.concatenate(inputs=[s_llp, g], axis=-1)                         
        # s_llp = keras.layers.merge.Concatenate(axis=-1)([s_llp, g])
        print ("state shape: ", s_llp)
        ### Decoding models
        ### https://github.com/keras-team/keras/issues/7949
        def repeat_vector(args):
            ### sequence_layer is used to determine how long the repetition should be
            layer_to_repeat = args[0]
            repeats = args[1]
            return RepeatVector(repeats)(layer_to_repeat)
                                                 
        # gen_states = repeat_vector((s_llp, self.getSettings()["hlc_timestep"]))
        stacked_goal = repeat_vector((g, self.getSettings()["hlc_timestep"]))
        # gen_states = keras.layers.Lambda(repeat_vector, output_shape=(None, keras.backend.int_shape(s_llp)[1])) ([s_llp, self.getSettings()["hlc_timestep"]])
        # gen_states = keras.backend.repeat(s_llp, self.getSettings()["hlc_timestep"])
        print ("stacked_goal:", stacked_goal)
        gen_states = keras.layers.concatenate(inputs=[s_llp, stacked_goal], axis=-1)      
        print ("Front Policy state shape: ", gen_states)           
        gen_actions = keras.layers.TimeDistributed(self._llp, input_shape=(None, 1, keras.backend.int_shape(s_llp)[1]))(gen_states)
        print ("Front Policy gen_actions shape2: ", gen_actions)           
        # gen_states = keras.layers.Reshape((keras.backend.int_shape(s_llp)[1], self.getSettings()["hlc_timestep"]))(gen_states)
        # gen_states = keras.backend.repeat_elements(s_llp, self.getSettings()["hlc_timestep"], axis=-1)
        
        # gen_actions = self._llp(inputs=[gen_states])
        ### Flatten stacked actions
        print("gen_actions: ", gen_actions)
        print("gen_actions2: ", keras.backend.int_shape(gen_actions))
        self._model._policy2 = keras.layers.Flatten()(gen_actions)
        # self._model._policy2 = keras.layers.Reshape((keras.backend.int_shape(gen_actions)*self.getSettings()["hlc_timestep"]))(gen_actions)
        # self._model._policy2 = keras.layers.Reshape((keras.backend.int_shape(gen_actions)*self.getSettings()["hlc_timestep"],))(gen_actions)
        # K.shape(sequence_layer)[1]
        print ("Front Policy output shape: ", self._model._policy2)
        
        self._qFunc = (self._model.getCriticNetwork()(
                        [self._model.getStateSymbolicVariable(),
                         self._model._policy2]))
        self._combined = Model(input=[self._model.getStateSymbolicVariable(),
                                      self._LLP_State], 
                            output=self._qFunc)
        
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['learning_rate']), 
                                    beta_1=np.float32(0.9), beta_2=np.float32(0.999), 
                                    epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0000001),
                                    amsgrad=False, clipvalue=1.0)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Clipping: ", sgd.decay)
            print("sgd, critic: ", sgd)
        self._combined.compile(loss=[neg_y], optimizer=sgd)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("combined qFun Net summary: ",  self._combined.summary())
            
        self._get_actor_loss = K.function([self._model.getStateSymbolicVariable()
                                                 # ,K.learning_phase()
                                                 ], [self._qFunc])
        
    def genLLPActions(self, states, g, target_net=False):
        # g = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])
        ### a pi(a|s,g)
        # s_llp = states[:,:-self.getSettings()["goal_slice_index"]] ### remove last 3 dimensions
        s_llp = states[:,:4] ### remove last 3 dimensions
                         
        s_llp = np.concatenate((s_llp, g), axis=-1)
        # s_llp = np.repeat(s_llp, self.getSettings()["hlc_timestep"], axis=1)
        if (target_net == True):
            a_llp = self._llp_T.predict(s_llp)
        else:
            a_llp = self._llp.predict(s_llp)
        a_llp = np.repeat(a_llp, self.getSettings()["hlc_timestep"], axis=1)
        # a_llp = np.reshape(a_llp, (-1,15))
        return a_llp
        
    def updateFrontPolicy(self, lowerPolicy):
        self._llp.set_weights(lowerPolicy._model._actor.get_weights())
        
    def getGrads(self, states, actions=None, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if ( alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=theano.config.floatX)
        grads = self._get_gradients([states])
        return grads
    
    def updateTargetModel(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print ("Updating target Model")
        """
            Target model updates
        """
        # return
        ## I guess it is okay to lerp the entire network even though we only really want to 
        ## lerp the value function part of the networks, the target policy is not used for anythings
        all_paramsA = self._model.getCriticNetwork().get_weights()
        all_paramsA1 = self._model1.getCriticNetwork().get_weights()
        all_paramsB = self._modelTarget.getCriticNetwork().get_weights()
        all_paramsB1 = self._modelTarget1.getCriticNetwork().get_weights()
        if ('target_net_interp_weight' in self.getSettings()):
            lerp_weight = self.getSettings()['target_net_interp_weight']
        else:
            lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        
        all_params = []
        all_params1 = []
        for paramsA, paramsA1, paramsB, paramsB1 in zip(all_paramsA, all_paramsA1, all_paramsB, all_paramsB1):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            params1 = (lerp_weight * paramsA1) + ((1.0 - lerp_weight) * paramsB1)
            all_params.append(params)
            all_params1.append(params1)
        self._modelTarget.getCriticNetwork().set_weights(all_params)
        self._modelTarget1.getCriticNetwork().set_weights(all_params1)
        
        all_paramsA_Act = self._model.getActorNetwork().get_weights()
        all_paramsB_Act = self._modelTarget.getActorNetwork().get_weights()
        
        all_params = []
        for paramsA, paramsB in zip(all_paramsA_Act, all_paramsB_Act):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        self._modelTarget.getActorNetwork().set_weights(all_params)
        
    def updateTargetModelCritic(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print ("Updating target critic Model")
        """
            Target model updates
        """
        # return
        ## I guess it is okay to lerp the entire network even though we only really want to 
        ## lerp the value function part of the networks, the target policy is not used for anythings
        all_paramsA = self._model.getCriticNetwork().get_weights()
        all_paramsA1 = self._model1.getCriticNetwork().get_weights()
        all_paramsB = self._modelTarget.getCriticNetwork().get_weights()
        all_paramsB1 = self._modelTarget1.getCriticNetwork().get_weights()
        if ('target_net_interp_weight' in self.getSettings()):
            lerp_weight = self.getSettings()['target_net_interp_weight']
        else:
            lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        
        all_params = []
        all_params1 = []
        for paramsA, paramsA1, paramsB, paramsB1 in zip(all_paramsA, all_paramsA1, all_paramsB, all_paramsB1):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            params1 = (lerp_weight * paramsA1) + ((1.0 - lerp_weight) * paramsB1)
            all_params.append(params)
            all_params1.append(params1)
        self._modelTarget.getCriticNetwork().set_weights(all_params)
        self._modelTarget1.getCriticNetwork().set_weights(all_params1)
        
    def updateTargetModelActor(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print ("Updating target Actor Model")
        """
            Target model updates
        """
        # return
        ## I guess it is okay to lerp the entire network even though we only really want to 
        ## lerp the value function part of the networks, the target policy is not used for anythings
        if ('target_net_interp_weight' in self.getSettings()):
            lerp_weight = self.getSettings()['target_net_interp_weight']
        else:
            lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        
        all_paramsA_Act = self._model.getActorNetwork().get_weights()
        all_paramsB_Act = self._modelTarget.getActorNetwork().get_weights()
        
        all_params = []
        for paramsA, paramsB in zip(all_paramsA_Act, all_paramsB_Act):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        self._modelTarget.getActorNetwork().set_weights(all_params)
    
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._model.getActorNetwork().get_weights()))
        params.append(copy.deepcopy(self._model1.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._model1.getActorNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget.getActorNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget1.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget1.getActorNetwork().get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        """
        for i in range(len(params[0])):
            params[0][i] = np.array(params[0][i], dtype=theano.config.floatX)
            """
        self._model.getCriticNetwork().set_weights(params[0])
        self._model.getActorNetwork().set_weights( params[1] )
        self._model1.getCriticNetwork().set_weights(params[2])
        self._model1.getActorNetwork().set_weights( params[3] )
        self._modelTarget.getCriticNetwork().set_weights( params[4])
        self._modelTarget.getActorNetwork().set_weights( params[5])  
        self._modelTarget1.getCriticNetwork().set_weights( params[6])
        self._modelTarget1.getActorNetwork().set_weights( params[7])  

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
        ## Easy fix for computing actor loss
        # diff = self._bellman_error2()
        # self._tmp_diff_shared.set_value(diff)
        
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
    def trainCritic(self, states, actions, rewards, result_states, falls, G_t=[[0]], p=1.0):
        
        # print ("actions: ", actions)
        # self.setData(states, actions, rewards, result_states, falls)
        ### get actions for target policy
        target_actions = self._modelTarget.getActorNetwork().predict(result_states, batch_size=states.shape[0])
        if "td3_apply_noise_llp" not in self.getSettings() or not self.getSettings()["td3_apply_noise_llp"]:
            target_actions = target_actions + np.clip(np.random.normal(loc=0, scale=self._noise_scale, size=target_actions.shape), -self._c, self._c)
        if not (self._llp is None):
            # llp_target_state = result_states[:,:7]
            # llp_target_state[:,-3:] = target_actions_n 
            # target_actions_n = self._llp.predict(llp_target_state)
            target_actions = self.genLLPActions(result_states, target_actions, target_net=True)
        if "td3_apply_noise_llp" in self.getSettings() and self.getSettings()["td3_apply_noise_llp"]:
            target_actions = target_actions + np.clip(np.random.normal(loc=0, scale=self._noise_scale, size=target_actions.shape), -self._c, self._c)

        ### Get next q value
        q_vals_b = self._modelTarget.getCriticNetwork().predict([result_states, target_actions], batch_size=states.shape[0])
        q_vals_b1 = self._modelTarget1.getCriticNetwork().predict([result_states, target_actions], batch_size=states.shape[0])
        q_vals = np.concatenate((q_vals_b, q_vals_b1), axis=1)
        q_vals_min = np.min(q_vals, axis=-1, keepdims=True)
        # print ("q_vals_b, q_vals_b1, min(q_vals_b,q_vals_b1)", np.concatenate((q_vals_b, q_vals_b1, q_vals_min), axis=1))
        ## Compute target values
        # target_tmp_ = rewards + ((self._discount_factor* q_vals_b )* falls)
        target_tmp_ = rewards + ((self._discount_factor * q_vals_min ))
        
        if ('anneal_learning_rate' in self.getSettings()
            and (self.getSettings()['anneal_learning_rate'] == True)):
            K.set_value(self._model.getCriticNetwork().optimizer.lr, np.float32(self.getSettings()['critic_learning_rate']) * p)
            # lr = K.get_value(self._model.getCriticNetwork().optimizer.lr)
            # print ("New critic learning rate: ", lr)
        
        loss = self._model.getCriticNetwork().fit([states, actions], target_tmp_,
                        batch_size=states.shape[0],
                        epochs=1,
                        verbose=False,
                        shuffle=False)
        
        loss1 = self._model1.getCriticNetwork().fit([states, actions], target_tmp_,
                        batch_size=states.shape[0],
                        epochs=1,
                        verbose=False,
                        shuffle=False)
        
        self.updateTargetModelCritic()
        loss = (loss.history['loss'][0] + loss1.history['loss'][0])/2.0
        return loss
        
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, 
                   exp_actions, G_t=[[0]], forwardDynamicsModel=None, p=1.0):
        # self.setData(states, actions, rewards, result_states, falls)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print("values: ", np.mean(self._q_val()* (1.0 / (1.0- self.getSettings()['discount_factor']))), " std: ", np.std(self._q_val()* (1.0 / (1.0- self.getSettings()['discount_factor']))) )
            print("Rewards: ", np.mean(rewards), " std: ", np.std(rewards), " shape: ", np.array(rewards).shape)
        # print("Policy mean: ", np.mean(self._q_action(), axis=0))
        loss = 0
        # loss = self._trainActor()
        if ('anneal_learning_rate' in self.getSettings()
            and (self.getSettings()['anneal_learning_rate'] == True)):
            K.set_value(self._combined.optimizer.lr, np.float32(self.getSettings()['learning_rate']) * p)
        
        ### The rewards are not used in this update, just a placeholder
        if not ("skip_policy_training" in self.getSettings() and self.getSettings()["skip_policy_training"]):
            if (self._llp is not None):
                llp_states = states[:,16:]
                score = self._combined.fit([states, llp_states], rewards,
                  epochs=1, batch_size=states.shape[0],
                  verbose=0
                  # callbacks=[early_stopping],
                  )
            else:
                score = self._combined.fit([states], rewards,
                      epochs=1, batch_size=states.shape[0],
                      verbose=0
                      # callbacks=[early_stopping],
                      )
            q_fun = -score.history['loss'][0]
        else:
            print("skipping policy training")
            q_fun = 0.0
        
        # q_fun = np.mean(self._trainPolicy(states))
        
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            # print("Actions mean:     ", np.mean(actions, axis=0))
            poli_mean = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])
            print("Policy mean: ", np.mean(poli_mean, axis=0))
            # print("Actions std:  ", np.mean(np.sqrt( (np.square(np.abs(actions - np.mean(actions, axis=0))))/1.0), axis=0) )
            # print("Actions std:  ", np.std((actions - self._q_action()), axis=0) )
            # print("Actions std:  ", np.std((actions), axis=0) )
            # print("Policy std: ", np.mean(self._q_action_std(), axis=0))
            # print("Mean Next State Grad grad: ", np.mean(next_state_grads, axis=0), " std ", np.std(next_state_grads, axis=0))
            # print("Mean ation grad: ", np.mean(action_grads, axis=0), " std ", np.std(action_grads, axis=0))
            print ("Actor Loss: ", q_fun)
        self.updateTargetModelActor()    
        return q_fun
        
    def train(self, states, actions, rewards, result_states):
        loss = self.trainCritic(states, actions, rewards, result_states)
        lossActor = self.trainActor(states, actions, rewards, result_states)
        return loss
    
    def predict(self, state, deterministic_=True, evaluation_=False, p=None, sim_index=None, bootstrapping=False):
        # print ("self._state_bounds shape ", np.array(self._state_bounds).shape)
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        action_ = scale_action(self._model.getActorNetwork().predict(state, batch_size=1), self._action_bounds)
        return action_
    
    def predict_std(self, state, deterministic_=True, p=1.0):
        state = norm_state(state, self._state_bounds)   
        state = np.array(state, dtype=self._settings['float_type'])
        
        # action_std = self._model.getActorNetwork().predict(state, batch_size=1)[:,self._action_length:] * (action_bound_std(self._action_bounds))
        action_std = self._q_action_std([state])[0] * action_bound_std(self._action_bounds)
        # print ("Policy std: ", action_std)
        return action_std * p
    
    def predict_target(self, state, deterministic_=True, evaluation_=False, p=None, sim_index=None, bootstrapping=False):
        # print ("self._state_bounds shape ", np.array(self._state_bounds).shape)
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        action_ = scale_action(self._modelTarget.getActorNetwork().predict(state, batch_size=state.shape[0]), self._action_bounds)
        return action_
    
    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        # return scale_reward(self._q_valTarget(), self.getRewardBounds())[0]
        poli_mean = self._model.getActorNetwork().predict(state, batch_size=1)
        value = (self._model.getCriticNetwork().predict([state, poli_mean] , batch_size=1)* action_bound_std(self.getRewardBounds())) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        
        # value = scale_reward(self._q_func(state), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        return value
        # return self._q_val()[0]
    
    def q_values(self, state):
        """
            For returning a vector of q values, state should already be normalized
        """
        state = np.array(state, dtype=self._settings['float_type'])
        # print ("Getting q_values: ", state)
        poli_mean = self._model.getActorNetwork().predict(state, batch_size=state.shape[0])
        value = self._model.getCriticNetwork().predict([state, poli_mean] , batch_size=state.shape[0])
        # print ("q_values: ", value)
        return value

    def q_values2(self, states):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        """
        if ( "use_centralized_critic" in self.getSettings()
             and (self.getSettings()["use_centralized_critic"] == True)):
            return np.zeros((states.shape[0], 1))
        """
        states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=self._settings['float_type'])
        # return scale_reward(self._q_valTarget(), self.getRewardBounds())[0]
        poli_mean = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])
        if ("policy_connections" in self.getSettings()
            and (any([self.getSettings()["agent_id"] == m[1] for m in self.getSettings()["policy_connections"]])) ):
            return np.zeros((states.shape[0],1))
            poli_mean = self.genLLPActions(states)
        value = (self._model.getCriticNetwork().predict([states, poli_mean] , 
                    batch_size=states.shape[0])* action_bound_std(self.getRewardBounds())) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        
        return value
        # return self._q_val()[0]
        
    def bellman_error(self, states, actions, rewards, result_states, falls):
        """
            Computes the one step temporal difference.
        """
        y_ = self._modelTarget.getCriticNetwork().predict([result_states, actions], batch_size=states.shape[0])
        target_ = rewards + ((self._discount_factor * y_))
        target_actions = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])
        if "td3_apply_noise_llp" not in self.getSettings() or not self.getSettings()["td3_apply_noise_llp"]:
            target_actions = target_actions + np.clip(np.random.normal(loc=0, scale=self._noise_scale, size=target_actions.shape), -self._c, self._c)
        if not (self._llp is None):
            # llp_target_state = result_states[:,:7]
            # llp_target_state[:,-3:] = target_actions_n 
            # target_actions_n = self._llp.predict(llp_target_state)
            target_actions = self.genLLPActions(result_states, target_actions)
        if "td3_apply_noise_llp" in self.getSettings() and self.getSettings()["td3_apply_noise_llp"]:
            target_actions = target_actions + np.clip(np.random.normal(loc=0, scale=self._noise_scale, size=target_actions.shape), -self._c, self._c)
        values = self._model.getCriticNetwork().predict([states, target_actions], batch_size=states.shape[0])
        bellman_error = target_ - values
        return bellman_error
        # return self._bellman_errorTarget()
        
    def get_actor_regularization(self):
        return self._get_actor_regularization([])
    
    def get_actor_loss(self, state, action, reward, nextState, advantage):
        return self._get_actor_loss([state])
    
    def get_critic_regularization(self):
        return self._get_critic_regularization([])
    
    def get_critic_loss(self, state, action, reward, nextState):
        return self._get_critic_loss([state, action, reward, nextState, 0])   

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
        self._model._actor.save(fileName+"_actor"+suffix, overwrite=True)
        self._model._critic.save(fileName+"_critic"+suffix, overwrite=True)
        if (self._modelTarget is not None):
            self._modelTarget._actor.save(fileName+"_actor_T"+suffix, overwrite=True)
            self._modelTarget._critic.save(fileName+"_critic_T"+suffix, overwrite=True)
            
        try:
            from keras.utils import plot_model
            ### Save model design as image
            plot_model(self._model._actor, to_file=fileName+"_actor"+'.svg', show_shapes=True)
            plot_model(self._model._critic, to_file=fileName+"_critic"+'.svg', show_shapes=True)
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
        self._model._actor = load_model(fileName+"_actor"+suffix)
        self._model._critic = load_model(fileName+"_critic"+suffix)
        if (self._modelTarget is not None):
            self._modelTarget._actor = load_model(fileName+"_actor_T"+suffix)
            self._modelTarget._critic = load_model(fileName+"_critic_T"+suffix)
            
        # self.compile()
        # self._model._actor_train = load_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        print ("critic self.getStateBounds(): ", self.getStateBounds()) 
        # self._result_state_bounds = np.array(hf.get('_result_state_bounds'))
        hf.close()