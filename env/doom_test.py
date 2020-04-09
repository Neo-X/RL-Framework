import gym
from gym.wrappers import ResizeObservation
#from .openAIGym import OpenAIGymEnv
#from .sim.OpenAIGymEnv import OpenAIGymEnv
import sys
import os
import json
sys.path.append("../")
sys.path.append("../env")
sys.path.append("../characterSimAdapter/")
sys.path.append("../simbiconAdapter/")
sys.path.append("../simAdapter/")
from sim.OpenAIGymEnv import OpenAIGymEnv
import copy
import numpy as np

try:
    import ppaquette_gym_doom
    from ppaquette_gym_doom.wrappers.action_space import ToBox
    from ppaquette_gym_doom.wrappers.observation_space import FlattenScaleObservation

except:
    print("Doom not installed")
    pass
env_name = 'ppaquette/DoomDefendCenter-v0'
env = gym.make(env_name)
env = ToBox('minimal')(env)
env = ResizeObservation(env, (16, 16))
env = FlattenScaleObservation(env)
# conf = copy.deepcopy(settings)
# conf['render'] = render
# exp = OpenAIGymEnv(env, {})

rewards = []
states = []
time_limit=256
for i_episode in range(20):
    observation = env.reset()
    for t in range(time_limit):
        #env.render()
        print(observation)
        action = env.action_space.sample()
        # action = action * 0.0
        print ("action: ", action)
        observation, reward, done, info = env.step(action)
        print("Reward: ", reward)
        rewards.append(reward)
        states.append(observation)
        if (t >= (time_limit-1)) or done:
        # if (t >= (time_limit-1)):
            print("Episode finished after {} timesteps".format(t+1))
            print("mean reward: ", np.mean(rewards))
            print("std reward: ", np.std(rewards))
            break
        
print("mean reward: ", np.mean(rewards))
print("std reward: ", np.std(rewards))
print("reward min: ", np.min(rewards), " max ", np.max(rewards))
print("state mean - std: ", repr(np.mean(states, axis=0) - np.std(states, axis=0)) )
print("state mean + std: ", repr(np.mean(states, axis=0) + np.std(states, axis=0)) )
print("state std", repr(np.std(states, axis=0)))

print("")
print("min state: ", repr(np.min(states, axis=0)) )
print("max state: ", repr(np.max(states, axis=0)) )

