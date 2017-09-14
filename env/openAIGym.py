import gym
from gym import wrappers
from gym import envs
import roboschool
import numpy as np
# print(envs.registry.all())
# env = gym.make('CartPole-v0')
# env = gym.make('BipedalWalker-v2')
# import roboschool, gym; print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))
print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))
env = gym.make('RoboschoolHopper-v1')
# env = gym.make('Hopper-v1')
# env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')

print( "Action Space: ", env.action_space)
print( "Action Space high: ", env.action_space.high)
print( "Action Space low: ", env.action_space.low)
print( "State Spance: ", env.observation_space)
print( "State Spance high: ", env.observation_space.high)
print( "State Spance low: ", env.observation_space.low)

rewards = []
states = []
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # print("Reward: ", reward)
        rewards.append(reward)
        states.append(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("mean reward: ", np.mean(rewards))
            print("mean reward: ", np.std(rewards))
            break
        
print("mean reward: ", np.mean(rewards))
print("std reward: ", np.std(rewards))
print("reward min: ", np.min(rewards), " max ", np.max(rewards))

print("min state: ", np.mean(states, axis=0) - np.std(states, axis=0))
print("max reward: ", np.std(states, axis=0) + np.std(states, axis=0))

