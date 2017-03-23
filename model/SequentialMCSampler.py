import numpy as np
import lasagne
import sys
sys.path.append('../')
sys.path.append("../characterSimAdapter/")
from model.ModelUtil import *
import math
import heapq
import copy

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.Sampler import Sampler
from model.ForwardDynamicsSimulator import ForwardDynamicsSimulator
from actor.ActorInterface import *

class Sample(object):
    
    def __init__(self, val, data):
        self._val = val
        self._data = data
        
    def __cmp__(self, other):
        return cmp(self._val, other._val)
    
    def __eq__(self, other):
        return self._val == other._val

class SequentialMCSampler(Sampler):
    """
        This model using a forward dynamics network to compute the reward directly
    """
    def __init__(self, exp, look_ahead, settings):
        super(SequentialMCSampler,self).__init__(settings)
        self._look_ahead=look_ahead
        self._exp=exp
        
    def setEnvironment(self, exp):
        self._exp = exp
        self._fd.setEnvironment(exp)
    
    def sampleModel(self, model, forwardDynamics, current_state):
        print ("Starting SMC sampling")
        state__ = self._exp.getEnvironment().getSimState()
        _bestSample = self._sampleModel(model, forwardDynamics, current_state, self._look_ahead)
        self._exp.getEnvironment().setSimState(state__)
        self._bestSample = _bestSample
        return _bestSample
    
    def _sampleModel(self, model, forwardDynamics, current_state, look_ahead):
        """
            The current state in this case is a special simulation state not the same as the
            input states used for learning. This state can be used to create another simulation environment
            with the same state.
        
        """
        # import characterSim
        _bestSample=[[0],[-10000000], [], []]
        self._samples=[]
        current_state_copy = current_state 
        if isinstance(forwardDynamics, ForwardDynamicsSimulator):
            # current_state_copy = characterSim.State(current_state.getID(), current_state.getParams())
            current_state_copy = current_state
        # print ("Suggested Action: " + str(action) + " for state: " + str(current_state_copy))
        _action_params = []
        samples = []
        if self.getSettings()["use_actor_policy_action_suggestion"]:
            variance____=0.03
            variance__=[variance____]
            current_state_copy2 = copy.deepcopy(current_state_copy)
            ## Get first action
            for i in range(look_ahead):
                if isinstance(forwardDynamics, ForwardDynamicsSimulator):
                    current_state_copy3 = copy.deepcopy(current_state_copy2)
                    current_state_copy__ = self._exp.getEnvironment().getStateFromSimState(current_state_copy)
                    # print ("current_state_copy__: ", current_state_copy__)
                    pa = model.predict(np.array([current_state_copy__]))
                    if self.getSettings()["use_actor_policy_action_variance_suggestion"]:
                        
                        lSquared =(4.1**2)
                        ## This uses the learned model so in the case the state given must be that used by the model
                        current_state_copy3__ = self._exp.getEnvironment().getStateFromSimState(current_state_copy3)
                        variance__ = getModelPredictionUncertanty(model, current_state_copy3__, 
                                                        length=4.1, num_samples=32, settings=self.getSettings())
                        
                        variance__ = list(variance__) * look_ahead # extends the list for the number of states to look ahead
                        # print (var_)
                        if not all(np.isfinite(variance__)): # lots of nan values for some reason...
                            print ("Problem computing variance from model: ", )
                            print ("State: ", current_state_copy3__, " action: ", pa)
                            for fg in range(len(samp_)):
                                print ("Sample ", fg, ": ", samp_[fg], " Predictions: ", predictions_[fg])
                                
                        print ("Predicted Variance: " + str(variance__))
                    else:
                        variance__=[variance____]*(len(pa)*look_ahead)
                else:
                    variance__=[variance____]*len(action)
                    pa = model.predict(current_state_copy2)
                
                action = pa
                _action_params.extend(action)
                # print ("Suggested Action: " + str(action) + " for state: " + str(current_state_copy) + " " + str(current_state_copy.getParams()) + " with variance: " + str(variance__))
                current_state_copy3 = forwardDynamics._predict(state__c=current_state_copy2, action=pa)
                # samples = self.generateSamples(self._pol._action_bounds,  num_samples=5)
                # samples = self.generateSamples(bounds,  num_samples=self._settings["num_uniform_action_samples"])
            # num_samples_ = pow(self.getSettings()["num_uniform_action_samples"], self._pol._action_length)
            num_samples_ = self.getSettings()["num_uniform_action_samples"] * self._pol._action_length
            # print ("Number of initial random samples: ", num_samples_)
            samples = self.generateSamplesFromNormal(mean=_action_params, num_samples=num_samples_, variance_=variance__)
        else:
            samples = self.generateSamples(self._pol._action_bounds,  num_samples=self.getSettings()["num_uniform_action_samples"], repeate=look_ahead)
        # print ("Current state sample: " + str(current_state_copy.getParams()))
        for sample in samples:
            pa = sample
            # print ("sample: " + str(sample))
            actions_ = chunks(sample, self._pol._action_length)
            actions=[]
            for chunk in actions_:
                act_ = clampAction(chunk, self._pol._action_bounds)
                actions.extend(act_)
            # self.updateSampleWeights()
            actions=chunks(actions, self._pol._action_length)
            
            y=[]
            init_states=[]
            predictions=[]
            if isinstance(forwardDynamics, ForwardDynamicsSimulator):
                current_state_ = copy.deepcopy(current_state_copy)
                # actions = chunks(sample, self._pol._action_length)
                for act_ in actions:
                    current_state__ = self._exp.getEnvironment().getStateFromSimState(current_state_)
                    init_states.append(current_state__)
                    (prediction, reward__) = forwardDynamics._predict(state__c=current_state_, action=act_)
                    prediction_ = self._exp.getEnvironment().getStateFromSimState(prediction)
                    predictions.append(prediction_)
                    # print ("Current State: ", current_state_.getParams(), " Num: ", current_state_.getID())
                    # print ("Prediction: ", prediction.getParams(), " Num: ", prediction.getID())
                    # print ("Executed Action: ", act_)
                    ## This reward function is not going to work anymore
                    y.append(reward__)
                    current_state_ = copy.deepcopy(prediction)
                    # goalDistance(np.array(current_state_.getParams()), )
                    # print ("Y : " + str(y))
                    
            else:
                current_state_=current_state_copy
                # actions = chunks(sample, self._pol._action_length)
                for act_ in actions:
                    init_states.append(current_state_)
                    prediction = forwardDynamics.predict(state=current_state_, action=act_)
                    predictions.append(prediction)
                    y.append(reward(current_state_, prediction))
                    current_state_ = prediction
            # print (pa, y, id(y))
            if all(np.isfinite(y)): # lots of nan values for some reason...
                # print ("Good sample:")
                self.pushSample(sample, self.discountedSum(y))
            else : # this is bad, usually means the simulation has exploded...
                print ("Y: ", y, " Sample: ", sample)
                # self._fd.initEpoch(self._exp)
                
                
            if self.discountedSum(y) > self.discountedSum(_bestSample[1]):
                _bestSample[1] = y
                _bestSample[0] = pa[:self._pol._action_length]
                _bestSample[2] = init_states
                _bestSample[3] = predictions
            del y
        
        self.updateSampleWeights()
        print ("Starting Importance Sampling: *******")
        # print ("Current state sample: " + str(current_state_copy.getParams()))
        for i in range(self.getSettings()["adaptive_samples"]): # 100 samples from pdf
            # print ("Data probabilities: " + str(self._data[:,1]))
            # print ("Data rewards: " + str(self._data[:,0]))
            sample = self.drawSample()
            actions_ = chunks(sample, self._pol._action_length)
            actions=[]
            for chunk in actions_:
                act_ = clampAction(chunk, self._pol._action_bounds)
                actions.extend(act_)
            self.updateSampleWeights()
            actions=chunks(actions, self._pol._action_length)
            # print ("Action samples: " + str(list(actions)))
            """
            for item in self._samples:
                if all(item[1][0] == sample): # skip item already contained in samples
                    print ("Found duplicate***")
                    continue
            """
            pa = sample
            # print ("sample: " + str(sample))
            y=[]
            init_states=[]
            predictions=[]
            if isinstance(forwardDynamics, ForwardDynamicsSimulator):
                current_state_ = copy.deepcopy(current_state_copy)
                # actions = chunks(sample, self._pol._action_length)
                for act_ in actions:
                    current_state__ = self._exp.getEnvironment().getStateFromSimState(current_state_)
                    init_states.append(current_state__)
                    (prediction, reward__) = forwardDynamics._predict(state__c=current_state_, action=act_)
                    prediction_ = self._exp.getEnvironment().getStateFromSimState(prediction)
                    predictions.append(prediction_)
                    # print ("Current State: ", current_state_.getParams(), " Num: ", current_state_.getID())
                    # print ("Prediction: ", prediction.getParams(), " Num: ", prediction.getID())
                    # print ("Executed Action: ", act_)
                    ## This reward function is not going to work anymore
                    y.append(reward__)
                    current_state_ = copy.deepcopy(prediction)
                    # goalDistance(np.array(current_state_.getParams()), )
                    # print ("Y : " + str(y))
                    
            else:
                current_state_=current_state_copy
                # actions = chunks(sample, self._pol._action_length)
                for act_ in actions:
                    init_states.append(current_state_)
                    prediction = forwardDynamics.predict(state=current_state_, action=act_)
                    predictions.append(prediction)
                    y.append(reward(current_state_, prediction))
                    current_state_ = prediction
                    
            # print (pa, y)
            if ( np.all(np.isfinite(y)) and (np.all(np.greater(y, -10000.0))) and (np.all(np.less(y, 10000.0))) ): # lots of nan values for some reason...
                # print ("Good sample:")
                self.pushSample(sample, self.discountedSum(y))
                if self.discountedSum(y) > self.discountedSum(_bestSample[1]):
                    _bestSample[1] = y
                    _bestSample[0] = pa[:self._pol._action_length]
                    _bestSample[2] = init_states
                    _bestSample[3] = predictions
            else : # this is bad, usually means the simulation has exploded...
                print ("Y: ", y, " Sample: ", sample)
                self._fd.initEpoch(self._exp)
        _bestSample[1] = self.discountedSum(_bestSample[1])
        # print ("Best Sample: ", _bestSample[0], _bestSample[1])
        return _bestSample
    
    def discountedSum(self, rewards, discount_factor=0.7):
        """
            Assumed first reward was earliest
        """
        discounted_sum=0
        for state_num in range(len(rewards)):
            discounted_sum += (math.pow(discount_factor,state_num) * rewards[state_num])
        return discounted_sum
    
    def predict(self, state, evaluation_=False):
        """
            Returns the best action
        """
        ## hacky for now
        if ( not evaluation_ ):
            if isinstance(self._fd, ForwardDynamicsSimulator):
                # print ( "SMC exp: ", self._exp)
                # self._fd.initEpoch(self._exp)
                # state = self._exp.getEnvironment().getState()
                state_ = self._exp.getEnvironment().getSimState()
            
            self.sampleModel(model=self._pol, forwardDynamics=self._fd, current_state=state_)
            action = self.getBestSample()
            self._exp.getEnvironment().setSimState(state_)
            # if isinstance(self._fd, ForwardDynamicsSimulator):
            #     self._fd._sim.getEnvironment().setState(state)
            # print ("Best Action SMC: " + str(action))
            action = action[0]
            return action
        else:
            return super(SequentialMCSampler, self).predict(state, evaluation_=evaluation_)
    
    def pushSample(self, action, val):
        # print ("Val: " + str(val))
        # print ("[action,val]: " + str([action,val]))
        
        # print ("Samples: " )
        # print ("\n\n\n")
        samp = Sample(val, action)
        heapq.heappush(self._samples, samp)
        
    
    def updateSampleWeights(self):
        num_samples=self.getSettings()["num_adaptive_samples_to_keep"]
        if num_samples > len(self._samples):
            num_samples = len(self._samples)
        data_ = heapq.nlargest(num_samples, self._samples)
        data = []
        for item in data_:
            data.append([item._data, item._val])
        # data = list(data)
        # print ("Data: " + str(data))
        data = np.array(data)
        # print ("Largest N: " + str(data[:,1]))
        min = np.min(data[:,1], 0)
        max = np.max(data[:,1], 0)
        diff = max-min
        if 0.0 == diff:
            print ("Diff contains zero: " + str(diff))
            print ("Data, largets N: " + str(data[:,1]))
        data_ = (data[:,1]-min)/(diff)
        # data_ = data[:,1]-min
        sum = np.sum(data_, 0) ## To prevent division by 0
        weights = data_ / sum
        self._data = copy.deepcopy(data)
        # print ("Weights: " + str(weights))
        # print ("Data: " + str(self._data))
        self._data[:,1] = np.array(weights, dtype='float64')
        # Delete old samples
        # self._samples = []
        # print ("Done computing pdf data: " + str(self._data))
        
    
    def drawSample(self):
        samp = np.random.choice(self._data[:,0], p=np.array(self._data[:,1], dtype='float64'))
        # samp = np.random.choice(self._data[:,0])
        # print ("Sample: " + str(samp))
        # print ("Sample type: " + str(samp[0].dtype))
        samples = self.generateSamplesFromNormal(samp, 1, variance_=0.005)
        return samples[0]
    