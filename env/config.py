

SIMULATION_ENVIRONMENTS = """
{
"comment__": "These environments are great for evalauting new RL algorithms and using for learning how to do deepRL",
"NavGame_2D-v0": 
{
    "config_file": "./args/genBiped2D/biped2dfull_flat_with_terrain_features.txt",
    "time_limit": 256,
    "sim_name": "NavGame",
        "comment__": "Possible state bounds to be used for scaling states for networks",
    "state_bounds": [[ -10.0, -10.0],
                       [   10.0,  10.0]],
        "comment__": "Action scaling values to be used to scale values for the network",
    "action_bounds": [[-1.2, -1.2],
                      [ 1.2,  1.2]]
},
"NavGame_5D-v0": 
{
    "config_file": "./args/genBiped2D/biped2dfull_incline_with_terrain_features.txt",
    "time_limit": 256,
    "sim_name": "NavGame",
        "comment__": "Possible state bounds to be used for scaling states for networks",
    "state_bounds": [[ -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                       [   10.0,  10.0,  10.0,  10.0,  10.0,  10.0,  10.0,  10.0,  10.0,  10.0 ]],
        "comment__": "Action scaling values to be used to scale values for the network",
    "action_bounds": [[-1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2],
                      [ 1.2,  1.2,  1.2,  1.2,  1.2,  1.2,  1.2,  1.2,  1.2,  1.2]]
},
"NavGame_10D-v0": 
{
    "config_file": "./args/genBiped2D/biped2dfull_incline_with_terrain_features.txt",
    "time_limit": 256,
    "sim_name": "NavGame",
        "comment__": "Possible state bounds to be used for scaling states for networks",
    "state_bounds": [[ -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                       [   10.0,  10.0,  10.0,  10.0,  10.0,  10.0,  10.0,  10.0,  10.0,  10.0 ]],
        "comment__": "Action scaling values to be used to scale values for the network",
    "action_bounds": [[-1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2],
                      [ 1.2,  1.2,  1.2,  1.2,  1.2,  1.2,  1.2,  1.2,  1.2,  1.2]]
},
"ParticleGame_2D-v0": 
{
    "config_file": "./args/genBiped2D/biped2dfull_flat_with_terrain_features.txt",
    "time_limit": 256,
    "sim_name": "ParticleGame",
        "comment__": "Possible state bounds to be used for scaling states for networks",
    "state_bounds": [[ -10.0, -10.0],
                       [   10.0,  10.0]],
        "comment__": "Action scaling values to be used to scale values for the network",
    "action_bounds": [[-1.2, -1.2],
                      [ 1.2,  1.2]]
}
}
"""