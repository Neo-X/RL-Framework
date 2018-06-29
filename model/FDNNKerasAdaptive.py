
from model.DeepNNKerasAdaptive import DeepNNKerasAdaptive
import copy

class FDNNKerasAdaptive(DeepNNKerasAdaptive):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_, print_info=False):
        settings_ = copy.deepcopy(settings_)
        settings_['policy_network_layer_sizes'] = settings_['fd_network_layer_sizes']
        settings_['critic_network_layer_sizes'] = settings_['reward_network_layer_sizes']
        settings_['last_policy_layer_activation_type'] = settings_['last_fd_layer_activation_type']
        settings_['activation_type'] = settings_['reward_activation_type']
        settings_['policy_activation_type'] = settings_['fd_policy_activation_type']
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
        if ("use_dual_state_representations" in settings_
            and (settings_["use_dual_state_representations"] == True)):
            n_in = settings_["num_terrain_features"]
        super(FDNNKerasAdaptive,self).__init__(n_in, n_in, state_bounds, action_bounds, reward_bound, settings_, print_info=print_info)
        self._forward_dynamics_net = self._actor
        self._reward_net = self._critic

