
from model.DeepNNKerasAdaptive import DeepNNKerasAdaptive
import copy
import keras

class FDNNKerasAdaptive(DeepNNKerasAdaptive):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False, stateName="State", resultStateName="ResultState", **kwargs):
        n_out = n_in
        print ("before n_in: ", n_in)
        settings_ = copy.deepcopy(settings_)
        settings_['policy_network_layer_sizes'] = settings_['fd_network_layer_sizes']
        settings_['critic_network_layer_sizes'] = settings_['reward_network_layer_sizes']
        settings_['use_stochastic_policy'] = False
        if ("use_stochastic_policy_fd" in settings_):
            settings_['use_stochastic_policy'] = settings_['use_stochastic_policy_fd']
        if ("last_fd_layer_activation_type" in settings_):
            settings_['last_policy_layer_activation_type'] = settings_['last_fd_layer_activation_type']
        if ("reward_activation_type" in settings_):
            settings_['activation_type'] = settings_['reward_activation_type']
        if ("fd_policy_activation_type" in settings_):
            settings_['policy_activation_type'] = settings_['fd_policy_activation_type']
        if ("last_reward_activation_type" in settings_):
            settings_["last_critic_layer_activation_type"] = settings_["last_reward_activation_type"]
        if ("fd_network_dropout" in settings_
            and (settings_["fd_network_dropout"] > 0.001)):
            settings_['dropout_p'] = settings_["fd_network_dropout"]
            settings_["use_dropout_in_actor"] = True
        if ("use_stochastic_forward_dynamics" in settings_
                and (settings_['use_stochastic_forward_dynamics'] == True)):
            settings_['use_stochastic_policy'] = True
        ### For using two different state descriptions
        if ("fd_terrain_shape" in settings_):
            settings_["terrain_shape"] = settings_["fd_terrain_shape"]
        if ("fd_num_terrain_features" in settings_):
            settings_["num_terrain_features"] = settings_["fd_num_terrain_features"]
        if ("fd_network_leave_off_end" in settings_):
            settings_["network_leave_off_end"] = settings_["fd_network_leave_off_end"]
            
        if ("reward_network_leave_off_end" in settings_
            and (settings_["reward_network_leave_off_end"] == True)):
            settings_["critic_network_leave_off_end"] = settings_["reward_network_leave_off_end"]
        if ("use_dual_state_representations" in settings_
            and (settings_["use_dual_state_representations"] == True)):
            if (
                ("replace_next_state_with_pose_state" in settings_ 
                 and (settings_["replace_next_state_with_pose_state"] == True))
                or
                ("fd_use_multimodal_state" in settings_ 
                 and (settings_["fd_use_multimodal_state"] == True)
                 )
                ):
                if (("fd_use_full_multimodal_state" in settings_ and
                          (settings_["fd_use_full_multimodal_state"] == True))):
                    pass
                else: 
                    n_out = settings_["dense_state_size"]
            else:
                if ("append_camera_velocity_state" in settings_
                    and (settings_["append_camera_velocity_state"] == True)):
                    n_out = settings_["num_terrain_features"] + 2
                    n_in = n_out
                elif ("append_camera_velocity_state" in settings_
                    and (settings_["append_camera_velocity_state"] == "3D")):
                    n_out = settings_["num_terrain_features"] + 3
                    n_in = n_out
                else:
                    n_out = settings_["num_terrain_features"]
                
        self._Noise = keras.layers.Input(shape=(1,), name="Noise")
        
        if ("train_gan" in settings_
            and (settings_["train_gan"] == True)):
            settings_["train_gan"] = "yes"
            
        if ("train_LSTM_FD" in settings_ ):
            settings_["train_LSTM"] = settings_["train_LSTM_FD"]
        
        if ("train_LSTM_Reward" in settings_):
            settings_["train_LSTM_Critic"] = settings_["train_LSTM_Reward"]
            
        if ("train_LSTM_FD_stateful" in settings_ ):
            settings_["train_LSTM_stateful"] = settings_["train_LSTM_FD_stateful"]
        if ("fd_use_multimodal_state" in settings_ ):
            settings_["use_multimodal_state"] = settings_["fd_use_multimodal_state"]
                
        if ("using_encoder_decoder_fd" in settings_ ):
            settings_["using_encoder_decoder"] = settings_["using_encoder_decoder_fd"]
        if ("use_decoder_fd" in settings_ ):
            settings_["use_decoder"] = settings_["use_decoder_fd"]
            
        ### Don't pass this parameter to FD model    
        settings_["use_single_network"] = False
        
        print ("FD net n_out: ", n_out)
        print ("FD net n_in: ", n_in)
        super(FDNNKerasAdaptive,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=print_info, stateName=stateName, resultStateName=resultStateName, **kwargs)
        self._forward_dynamics_net = self._actor
        self._reward_net = self._critic


    def reset(self):
        if (isinstance(self._forward_dynamics_net, keras.models.Model)):
            self._forward_dynamics_net.reset_states()
            self._reward_net.reset_states()