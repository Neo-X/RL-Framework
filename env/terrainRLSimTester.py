# import gym

from simAdapter import terrainRLSim
# from OpenGL import GL
import numpy as np
# print(envs.registry.all())
# env = gym.make('CartPole-v0')
# env = gym.make('BipedalWalker-v2')
# import roboschool, gym; print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))
env = terrainRLSim.getEnv(env_name="PD_Humanoid_3D_WALK_Imitate_30FPS_v0", render=False)
# env.getEnv().setRender(True)
# env.init()
# env = gym.make('Hopper-v1')
# env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')

"""
print( "Action Space: ", env.action_space)
if (not isinstance(env.action_space, gym.spaces.Discrete)):
    print( "Action Space high: ", repr(env.action_space.high))
    print( "Action Space low: ", repr(env.action_space.low))
print( "State Space: ", env.observation_space)
if (not isinstance(env.observation_space, gym.spaces.Discrete)):
    print( "State Space high: ", repr(env.observation_space.high))
    print( "State Space low: ", repr(env.observation_space.low))
"""
actionSpace = env.getActionSpace()
env.setRandomSeed(1234)
    
rewards = []
states = []
time_limit=128
for i_episode in range(50):
    observation = env.reset()
    for t in range(time_limit):
        env.render()
        # print(observation)
        action = ((actionSpace.getMaximum() - actionSpace.getMinimum()) * np.random.uniform(size=actionSpace.getMinimum().shape[0])  ) + actionSpace.getMinimum()
        observation, reward, done, info = env.step([action])
        # print("Reward: ", reward)
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
print("state mean - std: ", repr(np.mean(states, axis=0) - np.std(states, axis=0)))
print("state mean + std: ", repr(np.mean(states, axis=0) + np.std(states, axis=0)))
print("state std", repr(np.std(states, axis=0)))

print("")
print("min state: ", np.min(states, axis=0))
print("max state: ", np.max(states, axis=0))

