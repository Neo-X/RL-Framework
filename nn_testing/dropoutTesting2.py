import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../learn/')
from model.ModelUtil import *
from model.DropoutNetwork import DropoutNetwork
from ExperienceMemory import ExperienceMemory
import math


def f(x):
    return (math.cos(x)-0.75)*(math.sin(x)+0.75)

def fNoise(x):
    out = f(x)
    if (x > 1.0) and (x < 2.0):
        # print "Adding noise"
        r = random.choice([0,1])
        if r == 1:
            out = x
        else:
            out = out
    return out

if __name__ == '__main__':
    
    state_bounds = np.array([[-5.0],[5.0]])
    action_bounds = np.array([[-4.0],[2.0]])
    reward_bounds = np.array([[-3.0],[1.0]])
    experience_length = 200
    batch_size=32
    # states = np.repeat([np.linspace(-5.0, 5.0, experience_length)],2, axis=0)
    states = np.linspace(-5.0,-2.0, experience_length/2)
    states = np.append(states, np.linspace(-1.0, 5.0, experience_length/2))
    old_states = states
    print states
    actions = np.array(map(fNoise, states))
    actionsNoNoise = np.array(map(f, states))
    settings = {}
    
    # states2 = np.transpose(np.repeat([states], 2, axis=0))
    # print states2
    model = DropoutNetwork(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, settings)
    print "Network model Loaded"
 
    experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), experience_length, continuous_actions=True)
    for i in range(experience_length):
        action_ = np.array([actions[i]])
        state_ = np.array([states[i]])
        # print "Action: " + str([actions[i]])
        experience.insert(norm_state(state_, state_bounds), norm_action(action_, action_bounds),
                           norm_state(state_, state_bounds), norm_reward(np.array([0]), reward_bounds))
    
    errors=[]
    for i in range(25000):
        _states, _actions, _result_states, _rewards = experience.get_batch(batch_size)
        # print _actions 
        error = model.train(_states, _actions)
        errors.append(error)
        # print "Error: " + str(error)
    
    
    states = np.linspace(-6.0, 6.0, experience_length)
    actionsNoNoise = np.array(map(f, states))
    
    predicted_actions = np.array(map(model.predict, states))
    predicted_actions_dropout = np.array(map(model.predictWithDropout, states))
    predicted_actions_var = []
    
    lSquared =0.001
    num_samples = 32
    modelPrecsionInv = (2*experience_length*1e-6 ) / (0.90) / lSquared
    predictions = []
    for i in range(len(states)):
        
        samp_ = np.repeat([states[i]],num_samples, axis=0)
        # print "Sample: " + str(samp_)
        for sam in samp_:
            predictions.append(model.predictWithDropout(sam))
        # print "Predictions: " + str(predictions)
        var_ = (modelPrecsionInv)+np.var(predictions)
        # print var_
        predicted_actions_var.append(var_)
        predictions=[]
    # predictions = model.predictWithDropout(samp_)
    predicted_actions_var = np.array(predicted_actions_var)
    # states=np.reshape(states, (experience_length,1))
    predicted_actions = np.reshape(predicted_actions, (experience_length,))
    print "states shape: " + str(states.shape)
    print "var shape: " + str(predicted_actions_var.shape)
    print "act shape: " + str(predicted_actions.shape)
    
    # print "var : " + str(predicted_actions_var)
    # print "act : " + str(predicted_actions)
    
    lower_var=[]
    upper_var=[]
    
    for i in range(len(states)):
        lower_var.append(predicted_actions[i]-predicted_actions_var[i])
        upper_var.append(predicted_actions[i]+predicted_actions_var[i])
     
    lower_var = np.array(lower_var)
    upper_var = np.array(upper_var)
    
