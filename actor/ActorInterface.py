from _ast import Or
import sys
from theano.scalar.basic import OR
sys.path.append("../characterSimAdapter/")

class ActorInterface(object):
    """
    _actions = [[0.6,-0.4,0.75],
                [0.6,-0.4,0.25],
                [0.6,-0.4,0.5],
                [0.8,-0.45,0.75],
                [0.8,-0.45,0.5],
                [0.8,-0.45,0.25],
                [0.2,-0.5,0.75],
                [0.2,-0.5,0.5],
                [0.2,-0.5,0.25]]
    """         
    
    def __init__(self, settings_, experience):
        self._settings = settings_
        self._actions = self._settings["discrete_actions"]
        # self._experience = experience
        self._reward_sum=0
        self._agent = None
        self._action_bounds = self._settings["action_bounds"]
        self._count = 0
        self._state_len = None
        
    def setEncoder(self, encoder):
        self._encoder = encoder
        
    def updateScalling(self, state, init=False):
        import numpy as np
        # print ("state: ", state)
        
        if (init):
            self._state_len = np.prod(state.shape)
            if ("replace_entropy_state_with_vae" in self._settings):
                self._state_mean =  np.zeros((1,self._settings["encoding_vector_size"]))
                self._state_var = np.ones_like(self._state_mean)
            else:
                self._state_mean =  state
                self._state_var = np.ones_like(state)
            return 
        
        if ("bayesian_mean_var" in self._settings):
            self._state_mean =  np.array(self._settings["bayesian_mean_var"]["mean"])
            self._state_var = np.array(self._settings["bayesian_mean_var"]["var"])
            return
        
        if (self._state_len is None):
            self._state_len = np.prod(state.shape)
        
        if ("replace_entropy_state_with_vae" in self._settings
            and (self._settings["replace_entropy_state_with_vae"] == True)):
            state = self._encoder.predict_encoding(state)
        elif ("replace_entropy_state_with_vae" in self._settings
              and (self._settings["replace_entropy_state_with_vae"] == "p(z)")):
            state = self._encoder.predict_encoding_z(state)
        else:    
            state = state[:,:self._state_len]
            
        # print ("self._state_len: ", self._state_len)
        if (self.count() == 1 
            or (self.count() == 0 )):
            self._state_mean =  state
            self._state_var = np.ones_like(state)
        else:
            x_mean_old = self._state_mean
            self._state_mean = self._state_mean + ((state - self._state_mean)/self.count())
            
        if ( self.count() == 2):
            self._state_var = (self._last_state - ((state+self._last_state)/2.0)**2)/2.0
            
        elif (self.count() > 2):
            self._state_var = (((self.count()-2)*self._state_var) + ((self.count()-1)*(x_mean_old - self._state_mean)**2) + ((state - self._state_mean)**2))
            self._state_var = (self._state_var/float(self.count()-1))
            
        self._state_var = np.fabs(self._state_var)
        self._last_state = state
        
    def entropyReward(self, state):
        import scipy.stats
        import numpy as np
        if ("replace_entropy_state_with_vae" in self._settings
            and (self._settings["replace_entropy_state_with_vae"] == True)):
            state = self._encoder.predict_encoding(state)
        elif ("replace_entropy_state_with_vae" in self._settings
              and (self._settings["replace_entropy_state_with_vae"] == "p(z)")):
            state = self._encoder.predict_encoding_z(state)
        else:
            state = state[:,:self._state_len]
        # ps = scipy.stats.norm(self._state_mean, self._state_var).pdf(state)
        # ps = (np.square(self._state_mean - state)/(2*(self._state_var)))  + (2 * np.pi * np.sqrt(self._state_var))
        ps = ((1.0/2) * np.log(2.0 * np.pi * np.sqrt(self._state_var)) + (np.square(self._state_mean - state)/(2*np.sqrt(self._state_var))))
        # print ("self._state_mean: ", repr(self._state_mean))
        # print ("self._state_var: ", repr(self._state_var))
        # ps = ps
        # print ("ps: ", ps)
        # r = np.prod(np.log(ps))
        r = - np.sum(ps)
        # print ("ps, r: ", r)
        return r
    
    def rewardICM(self, state, action, result_state):
        import scipy.stats
        import numpy as np
        result_state_hat = self._encoder.predict(state, action)
        r = np.sum(np.square(result_state_hat - result_state))
        return r
            
    def init(self):
        self._reward_sum=0
        
    def count(self):
        return self._count
        
    def initEpoch(self):
        self._reward_sum=0
        self._count = 0
        
    def hasNotFallen(self, exp):
        return 1
        
    def getActionParams(self, index):
        return self._actions[index]
    
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        import characterSim
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping=bootstrapping)
        
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        import characterSim
        action_ = action_[0]
        action_ = np.array(action_, dtype='float64')
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        action = characterSim.Action()
        # samp = paramSampler.generateRandomSample()
        action.setParams(action_)
        reward = exp.getEnvironment().act(action)
        if ( not np.isfinite(reward)):
            print ("Found some bad reward: ", reward, " for action: ", action_)
        self._reward_sum = self._reward_sum + reward
        return reward
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def setAgent(self, agent):
        self._agent = agent
        
    def updateActor(self, sim, action_):
        pass
        
        
def reward(previous_state, current_state):
    current_state = current_state[0]
    vel_dif  = np.abs(current_state[-1] - 1.0)
    reward = math.exp((vel_dif*vel_dif)*-2.0)
    return reward
"""
def reward(previous_state, current_state):
    index_start=3
    v1_x = previous_state[index_start]
    v3_x = current_state[index_start]
    v2_x = previous_state[index_start+2]
    v5_x = previous_state[index_start+4]
    v4_x = current_state[index_start+2]
    v8_x = v5_x - v2_x
    v7_x = v4_x - v8_x
    dx = v7_x - v3_x 
    
    index_start=4
    v1_y = previous_state[index_start]
    v3_y = current_state[index_start]
    v2_y = previous_state[index_start+2]
    v5_y = previous_state[index_start+4]
    v4_y = current_state[index_start+2]
    v8_y = v5_y - v2_y
    v7_y = v4_y - v8_y
    dy = v7_y - v3_y
    
    d = math.sqrt((dx*dx)+(dy*dy))
    return -d
   """ 
def travelDist(previous_state, current_state):
    index_start=3
    a0x_in_s0 = previous_state[index_start+2]
    a1x_in_s0 = previous_state[index_start+4]
    s0x_in_s0 = previous_state[index_start]
    s0x_in_s1 = current_state[index_start]
    a0x_in_s1 = current_state[index_start+2]
    dx = (a1x_in_s0-s0x_in_s0) - (a0x_in_s1-s0x_in_s1) 
    
    index_start=4
    a0y_in_s0 = previous_state[index_start+2]
    a1y_in_s0 = previous_state[index_start+4]
    s0y_in_s0 = previous_state[index_start]
    s0y_in_s1 = current_state[index_start]
    a0y_in_s1 = current_state[index_start+2]
    dy = (a1y_in_s0-s0y_in_s0) - (a0y_in_s1-s0y_in_s1) 
    
    d = math.sqrt((dx*dx)+(dy*dy))
    return d
    
def armDistFromTarget(previous_state):
    index_start=3
    a0x_in_s0 = previous_state[index_start+2]
    s0x_in_s0 = previous_state[index_start]
    x = (a0x_in_s0-s0x_in_s0)
    
    index_start=4
    a0y_in_s0 = previous_state[index_start+2]
    s0y_in_s0 = previous_state[index_start]
    y = (a0y_in_s0-s0y_in_s0)
    
    return np.array([x, y, 0])
    
def anchorDist(a0, a1, anchors):
    return np.array(anchors[a1]) - np.array(anchors[a0])
    
def goalDistance(current_state, anchors, goal_anchor):
    """
        Computes the Euclidean distance between the current state and the goal anchor 
    """ 
    current_state_ = current_state.getParams()
    # distance to current target anchor
    armDist = armDistFromTarget(current_state_)
    
    # distance from current target to goal
    anchor_dist = anchorDist(current_state.getID(), goal_anchor, anchors) 
    
    # total distance
    dx = armDist[0] + anchor_dist[0]
    dy = armDist[1] + anchor_dist[1]
    
    d = math.sqrt((dx*dx)+(dy*dy))
    return d
            
    
    
