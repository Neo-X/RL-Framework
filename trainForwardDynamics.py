import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from model.NeuralNetwork import NeuralNetwork
from util.ExperienceMemory import ExperienceMemory
import matplotlib.pyplot as plt
import math
from ModelEvaluation import *
from util.SimulationUtil import *
import time


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
    """
    State is the input state and Action is the desired output (y).
    """
    np.random.seed(23)
    settingsFileName = sys.argv[1]
    file = open(settingsFileName)
    settings = json.load(file)
    print ("Settings: " , str(json.dumps(settings)))
    file.close()
    settings = validateSettings(settings)
    anchor_data_file = open(settings["anchor_file"])
    _anchors = getAnchors(anchor_data_file)
    print ("Length of anchors epochs: ", str(len(_anchors)))
    anchor_data_file.close()
    train_forward_dynamics=True
    model_type= settings["model_type"]
    directory= getDataDirectory(settings)
    num_actions= settings["num_actions"]
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    num_states=settings["num_states"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    # max_reward=settings["max_reward"]
    reward_bounds = np.array([[-10.1],[0.0]])
    batch_size=settings["batch_size"]
    train_on_validation_set=settings["train_on_validation_set"]
    state_bounds = np.array(settings['state_bounds'])
    discrete_actions = np.array(settings['discrete_actions'])
    print ("Sim config file name: ", str(settings["sim_config_file"]))
    # c = characterSim.Configuration(str(settings["sim_config_file"]))
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    action_space_continuous=settings['action_space_continuous']
    # states2 = np.transpose(np.repeat([states], 2, axis=0))
    # print states2
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
    
    if action_space_continuous:
        experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True)
    else:
        experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
    file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"expBufferInit.hdf5"
    # experience.saveToFile(file_name)
    experience.loadFromFile(file_name)
    state_bounds = experience._state_bounds
    
    if (settings['train_forward_dynamics']):
        print ("Created forward dynamics network")
        # model = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
        model = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None)
        if settings['visulaize_forward_dynamics']:
            from NNVisualize import NNVisualize
            nlv = NNVisualize(title=str("Forward Dynamics Model") + " with " + str(settings["model_type"]))
            nlv.setInteractive()
            nlv.init()
    
    # experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), experience_length, continuous_actions=True)
    """
    for i in range(experience_length):
        action_ = np.array([actions[i]])
        state_ = np.array([states[i]])
        # print "Action: " + str([actions[i]])
        experience.insert(norm_state(state_, state_bounds), norm_action(action_, action_bounds),
                           norm_state(state_, state_bounds), norm_reward(np.array([0]), reward_bounds))
    """
    trainData = {}
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    trainData["mean_forward_dynamics_loss"]=[]
    trainData["std_forward_dynamics_loss"]=[]
    trainData["mean_eval"]=[]
    trainData["std_eval"]=[]
    # dynamicsLosses=[]
    best_dynamicsLosses=1000000
    _states, _actions, _result_states, _rewards = experience.get_batch(batch_size)
    """
    _states = theano.shared(np.array(_states, dtype=theano.config.floatX))
    _actions = theano.shared(np.array(_actions, dtype=theano.config.floatX))
    _result_states = theano.shared(np.array(_result_states, dtype=theano.config.floatX))
    _rewards = theano.shared(np.array(_rewards, dtype=theano.config.floatX))
    """
    model.setData(_states, _actions, _result_states)
    for round_ in range(rounds):
        t0 = time.time()
        for epoch in range(epochs):
            # _states, _actions, _result_states, _rewards = experience.get_batch(batch_size)
            # print _actions 
            # dynamicsLoss = model.train(states=_states, actions=_actions, result_states=_result_states)
            dynamicsLoss = model._train()
        t1 = time.time()
        if (settings['train_forward_dynamics']):
            dynamicsLoss_ = model.bellman_error(np.array(_states), np.array(_actions), np.array(_result_states))
            # dynamicsLoss_ = model.bellman_error((_states), (_actions), (_result_states))
            dynamicsLoss = np.mean(np.fabs(dynamicsLoss_))
            # dynamicsLosses.append(dynamicsLoss)
            mean_dynamicsLosses = dynamicsLoss
            std_dynamicsLosses = np.std((dynamicsLoss_))
            if (settings['train_forward_dynamics']):
                trainData["mean_forward_dynamics_loss"].append(mean_dynamicsLosses)
                trainData["std_forward_dynamics_loss"].append(std_dynamicsLosses)
            print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " ForwardPredictionLoss: " + str(dynamicsLoss) + " in " + str(t1-t0) + " seconds")
            if mean_dynamicsLosses < best_dynamicsLosses:
                best_dynamicsLosses = mean_dynamicsLosses
                print ("Saving BEST current forward dynamics model: " + str(best_dynamicsLosses))
                file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+"_Best.pkl"
                f = open(file_name_dynamics, 'wb')
                dill.dump(model, f)
                f.close()
        if (settings['visulaize_forward_dynamics']):
            nlv.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
            nlv.redraw()
            nlv.setInteractiveOff()
            nlv.saveVisual(directory+"trainingGraphNN")
            nlv.setInteractive()
        # print "Error: " + str(error)
    
    
    