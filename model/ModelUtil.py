
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
        out.append(actionV[i] + random.gauss(0, explorationRate))
    return out

def randomExporation(explorationRate, actionV, bounds):
    """
        This version scales the exploration noise wrt the action bounds
    """
    
    out = []
    for i in range(len(actionV)):
        out.append(actionV[i] + random.gauss(0, explorationRate * (bounds[1][i]-bounds[0][i])))
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
    # state = np.array(state, dtype=)
    action = model.predict(state)
    init_value = model.q_value(state)
    """
    fake_state_ = copy.deepcopy(state)
    for i in range(num_updates):
        fake_state_ = fake_state_ + ( model.getGrads(fake_state_)[0] * learning_rate )
        print ("Fake state Value: ", model.q_value(fake_state_))
    """
    # print ("Initial value: ", init_value)
    init_action = copy.deepcopy(action)
    for i in range(num_updates):
        ## find next state with dynamics model
        next_state = np.reshape(forwardDynamicsModel.predict(state, action), (1, model.getStateSize()))
        value_ = model.q_value(next_state)
        # print ("next state q value: ", value_)
        # print ("Next State: ", next_state.shape)
        ## compute grad for next state wrt model, i.e. how to change the state to improve the value
        next_state_grads = model.getGrads(next_state)[0] * (learning_rate) # this uses the value function
        # print ("Next State Grad: ", next_state_grads)
        # next_state_grads = np.sum(next_state_grads, axis=1)
        # print ("Next State Grad shape: ", next_state_grads.shape)
        ## modify next state wrt increasing grad, this is the direction we want the next state to go towards 
        next_state = next_state + next_state_grads
        # print ("Next State: ", next_state)
        value_ = model.q_value(next_state)
        # print ("Updated next state q value: ", value_)
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
        # print ("Final Estimated Value: ", final_value)
        
        # repeat
    print ("New action: ", action, " action diff: ", (action - init_action))
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
    if (not valid):
        print ("Bounds to small:", np.greater(diff, epsilon))
        
    return valid
    

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    
    settingsFileName = sys.argv[1]
    file = open(settingsFileName)
    settings = json.load(file)
    file.close()
    
    action_bounds = np.array(settings["action_bounds"], dtype=float)
    action_bounds[0][0] = 0
    reward_bounds = np.array([[-10.1],[0.0]])
    action = np.array([0.20])
    print ("Action bounds: " + str(action_bounds))
    print ("Action: " + str(action))
    print ("Normalized Action: " + str(norm_action(action, action_bounds)) + " same action: " + str(scale_action(norm_action(action, action_bounds), action_bounds)))
    print ("Normalized Action: " + str(norm_action(action+0.5, action_bounds)) + " same action: " + str(scale_action(norm_action(action+0.5, action_bounds), action_bounds)))
    print ("Normalized Action: " + str(norm_action(action+-0.5, action_bounds)) + " same action: " + str(scale_action(norm_action(action+-0.5, action_bounds), action_bounds)))
    actions_=[]
    for i in range(50):
        action_ = randomExporation(settings["exploration_rate"], action, action_bounds)
        print (" Exploration action: ", action_)
        actions_.append(action_[0])
    data = actions_
    plt.hist(data, bins=20)
    plt.show()
    reward=np.array([-9.0])
    print ("Norm Reward: " + str(norm_reward(reward, reward_bounds)) )
    print ("Norm Reward: " + str(norm_reward(reward+-0.5, reward_bounds)))
    
    print ("Valid bounds: " + str(validBounds([[ -6.34096220e-01,   9.42412945e-01,  -2.77025047e+00,
          5.99883344e-01,  -5.19683588e-01,  -5.19683588e-01,
         -3.42888109e-01,  -5.23556180e-01,   7.76794216e-02,
         -9.48665544e-02,  -1.25781827e+00,  -5.54537258e-01,
         -1.47478797e-02,   3.06775891e-01,  -5.49878858e-02,
         -3.10999480e-01,  -5.23225430e-01,   4.41216961e-02,
          6.70018120e-02,  -2.68502903e-01,  -1.07900884e-01,
         -3.31729491e-01,  -8.55080422e-01,  -4.32993609e-01,
         -1.05998050e-01,  -2.68106419e-01,  -1.07713842e-01,
         -2.50738648e-01,  -8.67029229e-01,  -4.42178656e-01,
         -2.98209530e-02,   6.47656790e-01,  -6.87410339e-02,
         -3.04862383e-01,  -5.17820447e-01,  -2.19130177e-02,
          1.87506175e-01,   2.88989499e-01,  -5.06674074e-02,
         -3.06572773e-01,  -5.28246094e-01,   9.33682034e-03,
         -2.12056797e-01,   2.88929341e-01,  -4.98415256e-02,
         -3.06224259e-01,  -5.28127163e-01,   1.84725992e-03,
         -5.74297924e-03,  -7.39800415e-01,  -3.28428126e-01,
         -2.37915139e-01,  -1.49736493e+00,  -8.38406722e-01,
         -1.22256339e-01,  -7.39480049e-01,  -3.29799656e-01,
         -2.03172964e-01,  -1.51351519e+00,  -8.77449749e-01,
          1.85604062e-01,  -1.00539563e-01,  -6.78502488e-02,
         -3.08624715e-01,  -5.32719181e-01,  -2.79023435e-01,
         -2.13983417e-01,  -1.00673412e-01,  -6.55732138e-02,
         -3.12425854e-01,  -5.33302855e-01,  -3.10067463e-01,
         -4.53427219e-02,  -1.00490585e+00,  -3.94977763e-01,
         -2.29839893e-01,  -1.30519213e+00,  -7.17686616e-01,
         -1.29773048e-01,  -1.00561552e+00,  -3.99412635e-01,
         -3.34232676e-01,  -1.31537288e+00,  -7.46435832e-01,
         -4.72069519e-02,  -1.02928355e+00,  -3.19346926e-01,
         -1.22069807e-01,  -4.60775992e-01,  -7.68291495e-01,
         -1.24429322e-01,  -1.02959452e+00,  -3.23858854e-01,
         -1.56321658e-01,  -4.63447258e-01,  -7.87323291e-01],
       [  5.52303145e-01,   1.04555761e+00,   3.02666000e+01,
          1.00002054e+00,   8.06735146e-01,   8.06735146e-01,
          3.51870125e-01,   9.91835992e-02,   1.46733007e+00,
          1.20645504e-01,   1.27079125e+00,   5.69349748e-01,
          1.47486517e-02,   4.01719900e-01,   3.42407443e-03,
          3.17251037e-01,   1.04200792e-01,   1.46485770e+00,
          1.07099820e-01,  -2.24500388e-01,   1.59571481e-01,
          2.57378948e-01,   1.38080353e-01,   1.54077880e+00,
         -6.78278735e-02,  -2.24941047e-01,   1.59509411e-01,
          3.43361385e-01,   1.41715459e-01,   1.53683745e+00,
          2.97725769e-02,   6.67872076e-01,   7.29232433e-02,
          3.09939696e-01,   1.23269819e-01,   1.36720212e+00,
          2.12060324e-01,   3.09593576e-01,   2.78002296e-02,
          3.12247037e-01,   1.11951187e-01,   1.48659232e+00,
         -1.87558614e-01,   3.09643560e-01,   2.65821725e-02,
          3.12895954e-01,   1.12183498e-01,   1.49365569e+00,
          1.25905392e-01,  -6.39025192e-01,   3.38600265e-01,
          1.98073892e-01,   4.55785665e-01,   1.35369655e+00,
          2.88702715e-03,  -6.39567527e-01,   3.43208581e-01,
          2.41450683e-01,   4.51482942e-01,   1.37950851e+00,
          2.13967233e-01,  -8.02279111e-02,   3.96722428e-02,
          3.18095279e-01,   1.06835040e-01,   1.63583511e+00,
         -1.85688622e-01,  -8.01561112e-02,   3.65994168e-02,
          3.14904357e-01,   1.08489792e-01,   1.66426497e+00,
          1.34669408e-01,  -9.01365767e-01,   3.74676050e-01,
          3.23646209e-01,   5.41831850e-01,   1.34261187e+00,
          4.15927916e-02,  -9.00870526e-01,   3.85834112e-01,
          2.30340241e-01,   5.35241615e-01,   1.38430484e+00,
          1.29370966e-01,  -9.27492425e-01,   4.43455684e-01,
          1.60730840e-01,   2.58810565e-01,   1.61350876e+00,
          4.36603207e-02,  -9.27366355e-01,   4.54681722e-01,
          1.24063643e-01,   2.61238771e-01,   1.64705196e+00]])))
    
    
    