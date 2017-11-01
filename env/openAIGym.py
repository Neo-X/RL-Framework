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
env = gym.make('Pendulum-v0')
# env = gym.make('Hopper-v1')
# env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')

print( "Action Space: ", env.action_space)
if (not isinstance(env.action_space, gym.spaces.Discrete)):
    print( "Action Space high: ", env.action_space.high)
    print( "Action Space low: ", env.action_space.low)
print( "State Space: ", env.observation_space)
if (not isinstance(env.observation_space, gym.spaces.Discrete)):
    print( "State Space high: ", env.observation_space.high)
    print( "State Space low: ", env.observation_space.low)

rewards = []
states = []
time_limit=128
for i_episode in range(20):
    observation = env.reset()
    for t in range(time_limit):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
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
print("state mean - std: ", np.mean(states, axis=0) - np.std(states, axis=0))
print("state mean + std: ", np.mean(states, axis=0) + np.std(states, axis=0))
print("state std", repr(np.std(states, axis=0)))

print("")
print("min state: ", np.min(states, axis=0))
print("max state: ", np.max(states, axis=0))

