import numpy as np
# import lasagne
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import math
import random
import json

f2_scale = 5.0


def f(x):
    return ((math.cos(x)-0.75)*(math.sin(x)+0.75))

def f2(x):
    return ((math.cos(x*2)-0.75)*(math.sin(x/2)+0.75))*f2_scale

def f3(x):
    return ((math.fabs(x)*2))

def fNoise(x):
    out = f(x)
    if (x > -1.0) and (x < 0.0):
        # print "Adding noise"
        # r = random.choice([0,1])
        n = np.random.normal(0, 0.2 * (np.abs(x)+1), 1)[0]
        out = out + n
    return out

def f2Noise(x):
    out = f2(x)
    if (x > -2.0) and (x < -1.0):
        # print "Adding noise"
        # r = random.choice([0,1])
       n = np.random.uniform(0, 1.2 * (np.abs(x)+1), 1)[0]
       out = out + n
    return out

def f3Noise(x):
    out = f3(x)
    # if (x > -2.0) and (x < -1.0):
        # print "Adding noise"
        # r = random.choice([0,1])
    #     n = np.random.normal(0, 1.2 * (np.abs(x)+1), 1)[0]
    #     out = out + n
    return out

def createData(bounds, num_samples):
    samples = []
    scaling = (bounds[1][0] - bounds[0][0]) / 2.0
    mean = (bounds[1][0] + bounds[0][0]) / 2.0
    while (len(samples) < num_samples):
        samp = (((np.random.rand()-0.5) * 2.0) * scaling ) + mean
        if (samp < -1 or (samp > 0)):
            samples.append(samp)
        print ("samp: ", samp)
            
    return samples

if __name__ == '__main__':
    
    file = open(sys.argv[1])
    settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings)))
    file.close()
    num_members = settings['fd_ensemble_size']
    
    f_function=f2
    f_function_noise=f2Noise

    from util.SimulationUtil import setupEnvironmentVariable, setupLearningBackend
    setupEnvironmentVariable(settings)
    setupLearningBackend(settings)        
    
    from model.ModelUtil import *
    from util.SimulationUtil import createForwardDynamicsModel, createRLAgent, createEnvironment
    from util.ExperienceMemory import ExperienceMemory
    
    state_bounds =  np.array([[-3.0],[3.0]])
    # action_bounds = np.array([[-6.0*f2_scale],[1.0*f2_scale]])
    action_bounds = np.array([[-3.0],[3.0]])
    reward_bounds = np.array([[-3.0],[3.0]])
    experience_length = 5000
    batch_size=64
    # states = np.repeat([np.linspace(-5.0, 5.0, experience_length)],2, axis=0)
    states = createData(state_bounds, num_samples=experience_length)
    old_states = states
    # print states
    actions = list(map(f_function_noise, states))
    allAction = actions
    actionsNoNoise = list(map(f_function, states))
    
    # states2 = np.transpose(np.repeat([states], 2, axis=0))
    # print states2
    # model = DropoutNetwork(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, settings)
    print ("Creating forward dynamics network")
    # forwardDynamicsModel = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
    model = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=None,
                                                      reward_bounds=reward_bounds)
    
    experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), experience_length, continuous_actions=True, settings=settings)
    experience.setStateBounds(state_bounds)
    experience.setRewardBounds(reward_bounds)
    experience.setActionBounds(action_bounds)
    experience.setSettings(settings)
    arr = list(range(experience_length))
    random.shuffle(arr)
    num_samples_to_keep=300
    given_actions=[]
    given_states=[]
    for i in range(experience_length):
        action_ = [[allAction[arr[i]]]]
        given_actions.append(action_)
        state_ = [[states[arr[i]]]]
        given_states.append(state_)
        # print "Action: " + str([actions[i]])
        experience.insert(state_, action_, state_,[[0]])
    
    experience._updateScaling()
    model.setStateBounds(experience.getStateBounds())
    model.setActionBounds(experience.getActionBounds())
    model.setRewardBounds(experience.getRewardBounds())
    errors=[]
    for i in range(5000):
        _states, _actions, _result_states, _rewards, _falls, _G_ts, exp_actions__, _advantage = experience.get_batch(batch_size)
        # print _actions
        # print("mean shape: ", np.array(_states).shape, 
        #       " min: ", np.min(_states),
        #      " max: ", np.max(_states))
        # dynamicsLoss = forwardDynamicsModel.train(_states, _actions, _result_states, _rewards) 
        error = model.train(_states, _states + 1, _actions, _rewards)
        errors.append(error)
        # print "Error: " + str(error)
    
    print("Done training")
    states = np.linspace(state_bounds[0][0]*2, state_bounds[1][0]*2, num_samples_to_keep)
    actionsNoNoise = list(map(f_function, states))
    predicted_actions = []
    predicted_actions_dropout = []
    predicted_actions_std = []
    predicted_actions_var = []
    for i in range(num_members):
        predicted_actions.append(model.predict( np.reshape([states], newshape=(num_samples_to_keep,1)), 
                                            np.reshape([states], newshape=(num_samples_to_keep,1)) + 1, i))
        predicted_actions_std.append(model.predict_std( np.reshape([states], newshape=(num_samples_to_keep,1)), 
                                            np.reshape([states], newshape=(num_samples_to_keep,1)) + 1, i))
        predicted_actions_dropout.append(model.predictWithDropout(np.reshape([states], newshape=(num_samples_to_keep,1)), 
                                                          np.reshape([states], newshape=(num_samples_to_keep,1)) + 1, i))
    
        predictions = []
        predicted_actions_var_ = []
        for i in range(len(states)):
            
            var_ = getFDModelPredictionUncertanty(model, state=np.reshape([states[i]], newshape=(1,1)), 
                                                  action=np.reshape([states[i]], newshape=(1,1)) * 0,
                                                   length=0.5, num_samples=128)[0]
            predicted_actions_var_.append(var_)
        
        predicted_actions_var.append(np.array(predicted_actions_var_))
    print("Done sampling variance")
    predicted_actions_var = np.array(predicted_actions_var)
    predicted_actions = np.array(predicted_actions)
    print ("var shape: " + str(np.array(predicted_actions_var).shape))
    print ("act shape: " + str(np.array(predicted_actions).shape))
    # print("given_actions: ", given_actions)
    std = 1.0
    # _fig, (_bellman_error_ax, _reward_ax, _discount_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
    _fig, (_bellman_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
    # _bellman_error, = _bellman_error_ax.plot(old_states, actions, linewidth=2.0, color='y', label="True function with noise")
    
    for i in range(num_members):
        _bellman_error, = _bellman_error_ax.plot(states, predicted_actions_dropout[i], linewidth=2.0, color='r', label="Estimated function with dropout")
        _bellman_error, = _bellman_error_ax.plot(states, predicted_actions[i], linewidth=2.0, color='g', label="Estimated function")
        _bellman_error, = _bellman_error_ax.plot(states, predicted_actions_std[i], linewidth=2.0, color='b', label="Estimated function STD")
        _bellman_error, = _bellman_error_ax.plot(states, predicted_actions_var[i][:,0], linewidth=2.0, label="Variance")
        _bellman_error_std = _bellman_error_ax.fill_between(states, predicted_actions[i][:,0] - predicted_actions_var[i][:,0],
                                                             predicted_actions[i][:,0] + predicted_actions_var[i][:,0], facecolor='green', alpha=0.25)
    
    _bellman_error, = _bellman_error_ax.plot(states, actionsNoNoise, linewidth=2.0, label="True function")
    _bellman_error = _bellman_error_ax.scatter(np.array(given_states)[:,0,0], np.array(given_actions)[:,0,0], label="Data trained on")
    plt.grid(b=True, which='major', color='black', linestyle='-')
    plt.grid(b=True, which='minor', color='g', linestyle='--')
    legend = _bellman_error_ax.legend(loc='lower left', shadow=True)
    _bellman_error_ax.set_ylabel("Absolute Error")
    # _bellman_error_std = _bellman_error_ax.fill_between(states, predicted_actions - predicted_actions_var,
    # 
    """                                                    predicted_actions + predicted_actions_var, facecolor='green', alpha=0.5)
    actionsNoNoise2 = np.array(list(map(f2, states))) 
    _bellman_error, = _bellman_error_ax2.plot(old_states, actions2, linewidth=2.0, color='y', label="True function with noise")
    _bellman_error, = _bellman_error_ax2.plot(states, predicted_actions_dropout2, linewidth=2.0, color='r', label="Estimated function with dropout")
    _bellman_error, = _bellman_error_ax2.plot(states, predicted_actions2, linewidth=2.0, color='g', label="Estimated function")
    _bellman_error, = _bellman_error_ax2.plot(states, actionsNoNoise2, linewidth=2.0, label="Action2")
    _bellman_error = _bellman_error_ax2.scatter(np.array(given_states)[:,0,0], np.array(given_actions)[:,0,1], label="Data trained on")
    _bellman_error_std = _bellman_error_ax2.fill_between(states, lower_var2, upper_var2, facecolor='green', alpha=0.25)
    _bellman_error, = _bellman_error_ax2.plot(states, predicted_actions_var2, linewidth=2.0, label="Variance")
    # _bellman_error_ax.set_title("True function")
    # Now add the legend with some customizations.
    legend = _bellman_error_ax2.legend(loc='lower right', shadow=True)
    plt.grid(b=True, which='major', color='black', linestyle='-')
    plt.grid(b=True, which='minor', color='g', linestyle='--')
    """

    """
    _reward, = _reward_ax.plot([], [], linewidth=2.0)
    _reward_std = _reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
    _reward_ax.set_title('Mean Reward')
    _reward_ax.set_ylabel("Reward")
    _discount_error, = _discount_error_ax.plot([], [], linewidth=2.0)
    _discount_error_std = _discount_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
    _discount_error_ax.set_title('Discount Error')
    _discount_error_ax.set_ylabel("Absolute Error")
    plt.xlabel("Iteration")
    """
    _title = "Training function"
    _fig.suptitle(_title, fontsize=18)
    
    _fig.set_size_inches(8.0, 4.5, forward=True)
    er = plt.figure(2)
    plt.plot(range(len(errors)), errors)
    
    print ("Max var: " + str(np.max(predicted_actions_var_, axis=0)))
    print ("Min var: " + str(np.min(predicted_actions_var_, axis=0)))
    
    # _fig.show()
    # er.show()
    plt.show()
    
    _fig.savefig("fd_Noise.svg")
    _fig.savefig("fd_Noise.png")