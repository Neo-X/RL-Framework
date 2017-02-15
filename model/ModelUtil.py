
import json
import random
import sys
# sys.path.append("../characterSimAdapter/")
# import characterSim
import numpy as np
import math
import copy 

anchors__name="anchors"
# replace string print ([a-z|A-Z|0-9|\"| |:|(|)|\+|_|,|\.|-|\[|\]|\/]*)
def getAnchors(anchorFile):

    s = anchorFile.read()
    data = json.loads(s)
    return data[anchors__name]

def saveAnchors(anchors, fileName):
    data={}
    data[anchors__name] = anchors
    data_ = json.dumps(data)
    f = open(fileName, "w")
    f.write(data_)
    f.close()
    
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]
    
def btVectorToNumpy(vec):
    return np.array([vec.x(), vec.y(), vec.z()])

def thompsonExploration(model, exploration_rate, state_):
    """
        This exploration technique uses thompson exploration
        This generates and action wrt variance of the model and the mean
        a = mean + var.sample()
         
    """
    pa = model.predict(state_)
    pa_var = model.predictWithDropout(state_)
    diff = pa_var - pa
    out = pa + (diff * exploration_rate)
    return out
    

def eGreedy(pa1, ra2, e):
    """
        epsilon greedy action select
        pa1 is best action from policy
        ra1 is the random action
        e is proabilty to select random action
        0 <= e < 1.0
    """
    r = random.random()
    if r < e:
        return ra2
    else:
        return pa1
    
def eOmegaGreedy(pa1, ra1, ra2, e, omega):
    """
        epsilon greedy action select
        pa1 is best action from policy
        ra1 is the noisy policy action action
        ra2 is the random action
        e is proabilty to select random action
        0 <= e < omega < 1.0
    """
    r = random.random()
    if r < e:
        return ra2
    elif r < omega:
        return ra1
    else:
        return pa1

def norm_reward(reward, reward_bounds):
    return norm_action(reward, reward_bounds)

def scale_reward(reward, reward_bounds):
    return scale_action(reward, reward_bounds)

def norm_state(state, max_state):
    return norm_action(action_=state, action_bounds_=max_state)

def scale_state(state, max_state):
    return scale_action(normed_action_=state, action_bounds_=max_state)

def norm_action(action_, action_bounds_):
    """
        
        Normalizes the action 
        Where the middle of the action bounds are mapped to 0
        upper bound will correspond to 1 and -1 to the lower
        from environment space to normalized space
    """
    avg = (action_bounds_[0] + action_bounds_[1])/2.0
    return (action_ - (avg)) / (action_bounds_[1]-avg)

def scale_action(normed_action_, action_bounds_):
    """
        from normalize space back to environment space
        Normalizes the action 
        Where 0 in the action will be mapped to the middle of the action bounds
        1 will correspond to the upper bound and -1 to the lower
    """
    avg = (action_bounds_[0] + action_bounds_[1])/2.0
    return normed_action_ * (action_bounds_[1] - avg) + avg

def getSettings(settingsFileName):
    file = open(settingsFileName)
    settings = json.load(file)
    # print ("Settings: " + str(json.dumps(settings)))
    file.close()
    
    return settings

def randomExporation(explorationRate, actionV):
    out = []
    for i in range(len(actionV)):
        out.append(actionV[i] + random.gauss(actionV[i], explorationRate))
    return out

def randomExporation(explorationRate, actionV, bounds):
    """
        This version scales the exploration noise wrt the action bounds
    """
    
    out = []
    for i in range(len(actionV)):
        out.append(actionV[i] + random.gauss(actionV[i], explorationRate * (bounds[1][i]-bounds[0][i])))
    return out

def randomUniformExporation(bounds):
    out = []
    for i in range(len(bounds[0])):
        out.append(np.random.uniform(bounds[0][i],bounds[1][i],1)[0])
    return out

def clampAction(actionV, bounds):
    """
    bounds[0] is lower bounds
    bounds[1] is upper bounds
    """
    for i in range(len(actionV)):
        if actionV[i] < bounds[0][i]:
            actionV[i] = bounds[0][i]
        elif actionV[i] > bounds[1][i]:
            actionV[i] = bounds[1][i]
    return actionV

def clampActionWarn(actionV, bounds):
    """
    bounds[0] is lower bounds
    bounds[1] is upper bounds
    Returns True if the actionV was outside the bounds
    """
    out=False
    for i in range(len(actionV)):
        if actionV[i] < bounds[0][i]:
            actionV[i] = bounds[0][i]
            out=True
        elif actionV[i] > bounds[1][i]:
            actionV[i] = bounds[1][i]
            out=True
    return (actionV, out)

"""
def initSimulation(settings):

    print ("Sim config file name: " + str(settings["sim_config_file"]))
    c = characterSim.Configuration(str(settings["sim_config_file"]))
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    action_space_continuous=settings['action_space_continuous']
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
        
    
    # this is the process that selects which game to play
    exp = characterSim.Experiment(c)
        
    
    if ( "Deep_NN2" == model_type):
        model = RLDeepNet(n_in=len(state_bounds[0]), n_out=num_actions, state_bounds=state_bounds, 
                          action_bounds=None, reward_bound=reward_bounds)
        train_forward_dynamics=False
    elif (model_type == "Deep_NN3" ):
        model = DeepRLNet3(n_in=len(state_bounds[0]), n_out=num_actions, state_bounds=state_bounds, 
                          action_bounds=None, reward_bound=reward_bounds)
        train_forward_dynamics=False
    elif (model_type == "Deep_CACLA" ):
        model = DeepCACLA(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds)
        omega = settings["omega"]
        exploration_rate = settings["exploration_rate"]
        train_forward_dynamics=True
    else:
        print ("Unknown model type: " + str(model_type))
    state = characterSim.State()
    
    exp.getActor().init()
    exp.getEnvironment().init()
    
    output={}
    output['exp']=exp
    output['model']=model
    
    return output
"""

def getOptimalAction(forwardDynamicsModel, model, state):
    """
        Computes the optimal action to be taken given
        the forwardDynamicsModel f and
        the value function (model) v
    """
    learning_rate=model.getSettings()['action_learning_rate']
    num_updates=1
    state_length = model.getStateSize()
    # print ("state_length ", state_length)
    # print ("State shape: ", state.shape)
    action = model.predict(state)
    init_value = model.q_value(state)
    """
    fake_state_ = copy.deepcopy(state)
    for i in range(num_updates):
        fake_state_ = fake_state_ + ( model.getGrads(fake_state_)[0] * learning_rate )
        print ("Fake state Value: ", model.q_value(fake_state_))
    """
    print ("Initial value: ", init_value)
    # init_action = copy.deepcopy(action)
    for i in range(num_updates):
        ## find next state with dynamics model
        next_state = np.reshape(forwardDynamicsModel.predict(state, action), (1, model.getStateSize()))
        value_ = model.q_value(next_state)
        print ("next state q value: ", value_)
        # print ("Next State: ", next_state.shape)
        ## compute grad for next state wrt model, i.e. how to change the state to improve the value
        next_state_grads = model.getGrads(next_state)[0] * (learning_rate * 0.1) # this uses the value function
        # print ("Next State Grad: ", next_state_grads)
        # next_state_grads = np.sum(next_state_grads, axis=1)
        # print ("Next State Grad shape: ", next_state_grads.shape)
        ## modify next state wrt increasing grad, this is the direction we want the next state to go towards 
        next_state = next_state + next_state_grads
        # print ("Next State: ", next_state)
        value_ = model.q_value(next_state)
        print ("Updated next state q value: ", value_)
        # Set modified next state as output for dynamicsModel
        # print ("Next State2: ", next_state)
        # compute grad to find
        # next_state = np.reshape(next_state, (model.getStateSize(), 1))
        # uncertanty = getModelValueUncertanty(model, next_state[0])
        # print ("Uncertanty: ", uncertanty)
        ## Compute the grad to change the input to produce the new target next state
        ## We will want to use the negative o this grad because the cost funtion is L2, the grad will make thid bigger, user - to pull action towards target action using this loss function 
        dynamics_grads = forwardDynamicsModel.getGrads(np.reshape(state, (1, model.getStateSize())), np.reshape(action, (1, model.getActionSize())), np.reshape(next_state, (1, model.getStateSize())))[0]
        # print ("action_grad1: ", action_grads)
        # print ("dynamics_grads size: ", dynamics_grads.shape)
        ## Grab the part of the grads that is the action
        action_grads = dynamics_grads[:, state_length:] * learning_rate 
        # action_grads = action_grads * learning_rate
        # print ("action_grad2: ", action_grads)
        ## Use grad to update action parameters
        action = action - action_grads
        action = action[0]
        # print ("action_grad: ", action_grads, " new action: ", action)
        # print ( "Action shape: ", action.shape)
        # print (" Action diff: ", (action - init_action))
        next_state_ = np.reshape(forwardDynamicsModel.predict(state, action), (1, model.getStateSize()))
        
        # print ("Next_state: ", next_state_.shape, " values ", next_state_)
        final_value = model.q_value(next_state_)
        print ("Final Estimated Value: ", final_value)
        
        # repeat
    # print ("New action: ", action, " action diff: ", (action - init_action))
    action = clampAction(action, model._action_bounds)
    return action

def getModelPredictionUncertanty(model, state, length=4.1, num_samples=32):
    """
        Computes the optimal action to be taken given
        the forwardDynamicsModel f and
        the value function (model) v
    """
    lSquared =(length**2)
    modelPrecsionInv = ((lSquared * (1.0 - model.getSettings()['dropout_p'])) / 
                        (2*model.getSettings()['expereince_length']*
                         model.getSettings()['regularization_weight'] ))**-1
    # print "Model Precision Inverse:" + str(modelPrecsionInv)
    predictions_ = []
    samp_ = np.repeat(np.array(state),num_samples, axis=0)
    # print "Sample: " + str(samp_)
    for pi in range(num_samples):
        predictions_.append( (model.predictWithDropout(samp_[pi])))
    # print "Predictions: " + str(predictions_)
    variance__ = (modelPrecsionInv) + np.var(predictions_, axis=0)
    return variance__

def getModelValueUncertanty(model, state, length=4.1, num_samples=32):
    """
        Computes the optimal action to be taken given
        the forwardDynamicsModel f and
        the value function (model) v
    """
    lSquared =(length**2)
    modelPrecsionInv = ((lSquared * (1.0 - model.getSettings()['dropout_p'])) / 
                        (2*model.getSettings()['expereince_length']*
                         model.getSettings()['regularization_weight'] ))**-1
    # print "Model Precision Inverse:" + str(modelPrecsionInv)
    predictions_ = []
    samp_ = np.repeat(np.array(state),num_samples, axis=0)
    # print "Sample: " + str(samp_)
    for pi in range(num_samples):
        predictions_.append( (model.q_valueWithDropout(samp_[pi])))
    # print "Predictions: " + str(predictions_)
    variance__ = (modelPrecsionInv) + np.var(predictions_, axis=0)
    return variance__


def validBounds(bounds):
    """
        Checks to make sure bounds are valid
        max is > min
    """
    valid = np.all(np.less(bounds[0], bounds[1]))
    if (not valid):
        print ("Invalid bounds: ", np.less(bounds[0], bounds[1]))
        
    # bounds not too close to each other
    epsilon = 0.01
    bounds = np.array(bounds)
    diff = bounds[1]-bounds[0]
    valid = valid and np.all(np.greater(diff, epsilon))
        
    return valid
    

if __name__ == '__main__':
    import sys
    
    settingsFileName = sys.argv[1]
    file = open(settingsFileName)
    settings = json.load(file)
    
    action_bounds = np.array(settings["action_bounds"], dtype=float)
    reward_bounds = np.array([[-10.1],[0.0]])
    action = np.array([0,0,0.0])
    print ("Action bounds: " + str(action_bounds))
    print ("Action: " + str(action))
    print ("Scaled Action: " + str(scale_action(action, action_bounds)) + " norm action: " + str(norm_action(scale_action(action, action_bounds), action_bounds)))
    print ("Scaled Action: " + str(scale_action(action+0.5, action_bounds)) + " norm action: " + str(norm_action(scale_action(action+0.5, action_bounds), action_bounds)))
    print ("Scaled Action: " + str(scale_action(action+-0.5, action_bounds)) + " norm action: " + str(norm_action(scale_action(action+-0.5, action_bounds), action_bounds)))
    
    reward=np.array([-9.0])
    print ("Norm Reward: " + str(norm_reward(reward, reward_bounds)) )
    print ("Norm Reward: " + str(norm_reward(reward+-0.5, reward_bounds)))
    
    
    