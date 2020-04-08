import numpy as np
import sys
sys.path.append('../')
sys.path.append("../characterSimAdapter/")
from model.ModelUtil import *
import math
import heapq
import copy
import os
# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.Sampler import Sampler
from model.ForwardDynamicsSimulator import ForwardDynamicsSimulator
from actor.ActorInterface import *



class Sample(object):
    
    def __init__(self, val, data):
        self._val = val
        self._data = copy.deepcopy(data)
        
    def __cmp__(self, other):
        return cmp(self._val, other._val)
    
    def __eq__(self, other):
        return self._val == other._val
    
    def __lt__(self, other):
        return self._val < other._val
    
    def __gt__(self, other):
        return self._val > other._val

class SequentialMCSampler(Sampler):
    """
        This model using a forward dynamics network to compute the reward directly
    """
    def __init__(self, exp, look_ahead, settings):
        super(SequentialMCSampler,self).__init__(settings)
        self._look_ahead=look_ahead
        self._exp=exp
        self._bad_reward_value=0
        self._previous_data=[]
        
    def setEnvironment(self, exp):
        self._exp = exp
        self._fd.setEnvironment(exp)
    
    def sampleModel(self, model, forwardDynamics, current_state):
        # print ("Starting SMC sampling: exp: ", self._exp )
        state__ = self._exp.getSimState()
        _bestSample = self._sampleModel(model, forwardDynamics, current_state, self._look_ahead)
        self._exp.setSimState(state__)
        self._bestSample = _bestSample
        return _bestSample
    
    def generateInitialSamples(self, model, forwardDynamics, current_state, look_ahead):

        _action_dimension = len(self.getSettings()["action_bounds"][0])
        _action_bounds = np.array(self.getSettings()["action_bounds"])
        if ("resuse_mbrl_samples" in self.getSettings()
            and "resuse_mbrl_samples" in self.getSettings()):
            self._previous_data
        current_state_copy = current_state 
        if isinstance(forwardDynamics, ForwardDynamicsSimulator):
            # current_state_copy = characterSim.State(current_state.getID(), current_state.getParams())
            current_state_copy = current_state
        # print ("Current State: " + str(current_state))
        _action_params = []
        samples = []
        if self.getSettings()["use_actor_policy_action_suggestion"]:
            variance____=self.getSettings()['variance_scalling']
            variance__=[]
            ### Start out at the same state for each trajectory
            current_state_copy2 = copy.deepcopy(current_state_copy)
            for i in range(look_ahead):
                if isinstance(forwardDynamics, ForwardDynamicsSimulator):
                    current_state_copy__ = self._exp.getStateFromSimState(current_state_copy)
                    pa = model.predict(np.array(current_state_copy__))
                else:
                    pa = model.predict(current_state_copy2)
                if (self.getSettings()["use_actor_policy_action_variance_suggestion"] == True):
                    
                    lSquared =(4.1**2)
                    ## This uses the learned model so in the case the state given must be that used by the model
                    current_state_copy3__ = self._exp.getStateFromSimState(current_state_copy2)
                    variance__.extend(getModelPredictionUncertanty(model, current_state_copy3__, 
                                                    length=4.1, num_samples=32, settings=self.getSettings())
                                      )
                    
                    # variance__ = list(variance__) * look_ahead # extends the list for the number of states to look ahead
                    # print (var_)
                    if not all(np.isfinite(variance__)): # lots of nan values for some reason...
                        print ("Problem computing variance from model: ", )
                        print ("State: ", current_state_copy3__, " action: ", pa)
                        for fg in range(len(samp_)):
                            print ("Sample ", fg, ": ", samp_[fg], " Predictions: ", predictions_[fg])
                            
                    print ("Predicted Variance: " + str(variance__))
                elif (self.getSettings()["use_actor_policy_action_variance_suggestion"] == "network"):
                    variance__.extend(model.predict_std(current_state_copy2)[0])
                else:
                    variance__.extend([variance____]*_action_dimension)
                
                action = pa
                _action_params.extend(action)
                ### progress forward in time
                if isinstance(forwardDynamics, ForwardDynamicsSimulator):
                    # print ("current_state_copy2: ", current_state_copy2)
                    (current_state_copy2, reward__) = forwardDynamics._predict(state__c=current_state_copy2, action=pa)
                else:
                    current_state_copy2 = forwardDynamics.predict(current_state_copy2, action)
            num_samples_ = self.getSettings()["num_uniform_action_samples"] * (_action_dimension)
            # print ("Number of initial random samples: ", num_samples_)
            _action_params = np.ravel(_action_params)
            # print ("_action_params: ", _action_params, " variance: ", variance__, " for pid: ", os.getpid())
            samples = self.generateSamplesFromNormal(mean=_action_params, num_samples=num_samples_, variance_=variance__)
        else:
            num_samples_ = self.getSettings()["num_uniform_action_samples"] * (_action_dimension)
            # samples = self.generateSamples(_action_bounds,  num_samples=self.getSettings()["num_uniform_action_samples"], repeate=look_ahead)
            samples = self.generateSamplesUniform(_action_bounds,  num_samples=num_samples_, repeate=look_ahead)
            # print ("Samples: ", samples)
        return samples
    
    def trySample(self, sample, model, forwardDynamics, current_state, look_ahead):
        
        mbrl_discount_factor = 0.7
        if ( "mbrl_discount_factor" in self.getSettings()):
            mbrl_discount_factor = self.getSettings()["mbrl_discount_factor"]
        sim_time = self._exp.getAnimationTime()
        _action_dimension = len(self.getSettings()["action_bounds"][0])
        _action_bounds = np.array(self.getSettings()["action_bounds"])
        # import characterSim
        current_state_copy = current_state
        
        pa = sample
        # print ("sample: " + str(sample))
        actions_ = chunks(sample, _action_dimension)
        actions=[]
        for chunk in actions_:
            # act__ = clampAction(chunk, _action_bounds)
            act__ = chunk
            actions.extend(act__)
        # self.updateSampleWeights()
        actions=list(chunks(actions, _action_dimension))
        
        y=[]
        init_states=[]
        predictions=[]
        if isinstance(forwardDynamics, ForwardDynamicsSimulator):
            current_state_ = copy.deepcopy(current_state_copy)
            # actions = chunks(sample, _action_dimension)
            forwardDynamics.setSimState(current_state_)
            for a in range(len(actions)):
                
                if ( ((not np.all(np.isfinite(actions[a])) or (np.any(np.less(actions[a], -10000.0))) or (np.any(np.greater(actions[a], 10000.0)))) or
                        forwardDynamics.endOfEpoch()  ) 
                     ): # lots of nan values for some reason...
                    ### Most likely end of epoch...
                    # print("Found bad action in search at: ", a, "actions[a]: ", actions[a])
                    ## Append bad values for the rest of the actions
                    y.append(self._bad_reward_value)
                    continue
                    # break
                
                current_state__ = self._exp.getStateFromSimState(current_state_)
                init_states.append(current_state__)
                (prediction, reward__) = forwardDynamics._predict(state__c=current_state_, action=[actions[a]])
                reward__ = reward__[0][0]
                # epochEnded = forwardDynamics.endOfEpoch()
                # print ("Epoch Ended: ", epochEnded, " on action: ", a)
                prediction_ = self._exp.getStateFromSimState(prediction)
                # print("prediction_: ", prediction_)
                # print ("(prediction, reward__): ", prediction, reward__)
                if ( ( not np.all(np.isfinite(prediction_))) or (np.any(np.less(prediction_, -10000.0))) or (np.any(np.greater(prediction_, 10000.0))) ): # lots of nan values for some reason...
                    print("Reached bad state in search")
                    # break
                
                    
                predictions.append(prediction_)
                # print ("Current State: ", current_state_.getParams(), " Num: ", current_state_.getID())
                # print ("Prediction: ", prediction.getParams(), " Num: ", prediction.getID())
                # print ("Executed Action: ", actions[a])
                ## This reward function is not going to work anymore
                if ( "use_state_based_reward_function" in self.getSettings() 
                     and (self.getSettings()["use_state_based_reward_function"] == True)):
                    reward__ = self._exp.computeReward(prediction_[0], sim_time + (a * 0.033))
                y.append(reward__)
                current_state_ = copy.deepcopy(prediction)
                # goalDistance(np.array(current_state_.getParams()), )
                # print ("Y : " + str(y))
                
        else:
            current_state_=current_state_copy
            # actions = chunks(sample, _action_dimension)
            for a in range(len(actions)):
                init_states.append(current_state_)
                if ("use_stochastic_forward_dynamics" in self.getSettings()
                    and (self.getSettings()["use_stochastic_forward_dynamics"])):
                    prediction = sampleStochasticModel( forwardDynamics, current_state_, [actions[a]])
                elif ("use_stochastic_gan" in self.getSettings()
                    and (self.getSettings()["use_stochastic_gan"])):
                    prediction = sampleStochasticGANModel( forwardDynamics, current_state_, [actions[a]])
                else:
                    prediction = forwardDynamics.predict(state=current_state_, action=[actions[a]])
                if ( not (np.all(np.isfinite(prediction)) and (np.all(np.greater(prediction, -10000.0))) and (np.all(np.less(prediction, 10000.0)))) ): # lots of nan values for some reason...
                    print("Reached bad state in search")
                    # break
                
                predictions.append(prediction)
                # print ("prediction: ", prediction)
                y.append(self._exp.computeReward(prediction[0], sim_time + (a * 0.033)))
                current_state_ = prediction
        # predictions__.append(predictions)
        # ys__.append(y)
        # print (pa, y, id(y))
        if ( np.all(np.isfinite(y)) and (np.all(np.greater(y, -10000.0))) and (np.all(np.less(y, 10000.0))) ): # lots of nan values for some reason...
            # print ("Good sample:")
            self.pushSample(sample, self.discountedSum(y, discount_factor=mbrl_discount_factor))
        else : # this is bad, usually means the simulation has exploded...
            print ("Y: ", y, " Sample: ", sample)
            print (" current_state_: ", current_state_copy)
            # self._fd.initEpoch(self._exp)
            # return _bestSample
            
            
        return (y, pa, init_states, predictions)
        
    def _sampleModel(self, model, forwardDynamics, current_state, look_ahead):
        """
            The current state in this case is a special simulation state not the same as the
            input states used for learning. This state can be used to create another simulation environment
            with the same state.
        
        """
        
        mbrl_discount_factor = 0.7
        if ( "mbrl_discount_factor" in self.getSettings()):
            mbrl_discount_factor = self.getSettings()["mbrl_discount_factor"]
        sim_time = self._exp.getAnimationTime()
        _action_dimension = len(self.getSettings()["action_bounds"][0])
        _action_bounds = np.array(self.getSettings()["action_bounds"])
        # import characterSim
        _bestSample=[[0],[-10000000], [], []]
        self._samples=[]
        current_state_copy = current_state
        
        samples = self.generateInitialSamples(model, forwardDynamics, current_state, look_ahead)
        
        predictions__ = []
        ys__ = []
        for sample in samples:
            (y, pa, init_states, predictions) = self.trySample(sample, model, forwardDynamics, current_state, look_ahead)
            if self.discountedSum(y, discount_factor=mbrl_discount_factor) > self.discountedSum(_bestSample[1], discount_factor=mbrl_discount_factor):
                _bestSample[1] = y
                _bestSample[0] = pa[:_action_dimension]
                _bestSample[2] = init_states
                _bestSample[3] = predictions
                # print ("samples: ", self._samples)
        
        for i in range(self.getSettings()["adaptive_samples"]): # 100 samples from pdf
            # print ("Data probabilities: " + str(self._data[:,1]))
            # print ("Data rewards: " + str(self._data[:,0]))
            self.updateSampleWeights()
            sample = self.drawSample()
            (y, pa, init_states, predictions) = self.trySample(sample, model, forwardDynamics, current_state, look_ahead)
            if self.discountedSum(y, discount_factor=mbrl_discount_factor) > self.discountedSum(_bestSample[1], discount_factor=mbrl_discount_factor):
                _bestSample[1] = y
                _bestSample[0] = pa[:_action_dimension]
                _bestSample[2] = init_states
                _bestSample[3] = predictions
        _bestSample[1] = self.discountedSum(_bestSample[1], discount_factor=mbrl_discount_factor)
        # print ("Best Sample: ", _bestSample[0], _bestSample[1])
        self._previous_data = self._samples
        return _bestSample
    
    def discountedSum(self, rewards, discount_factor=0.7):
        """
            Assumed first reward was earliest
        """
        discounted_sum=0
        for state_num in range(len(rewards)):
            discounted_sum += (math.pow(discount_factor,state_num) * rewards[state_num])
        return discounted_sum
    
    def predict(self, state, evaluation_=False, p=1.0, sim_index=None, bootstrapping=False):
        """
            Returns the best action
        """
        ## hacky for now
        if ( not evaluation_ ):
            if isinstance(self._fd, ForwardDynamicsSimulator):
                # print ( "SMC exp: ", self._exp)
                # self._fd.initEpoch(self._exp)
                # state = self._exp.getState()
                state = self._exp.getSimState()
                # self._exp.setSimState(state)
            # if ( self._exp.endOfEpoch() ):
            #     print ("Given back state where it is already endOfEpoch()")
            #     return self._pol.predict(state)
            
            self.sampleModel(model=self._pol, forwardDynamics=self._fd, current_state=state)
            action = self.getBestSample()
            # self._exp.setSimState(state)
            # if isinstance(self._fd, ForwardDynamicsSimulator):
            #     self._fd._sim.setState(state)
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
        if 0.0 == diff: ## To prevent division by 0
            ### Often happens when rewards are all 0 (robot fell)
            # print ("Diff contains zero: " + str(diff))
            # print ("Data, largest N: " + str(data[:,1]))
            # print ("Data, largest actions: " + str(data[:,0]))
            # for s in self._samples:
                # print ("self._samples[s]: ", s, s._val, s._data)
            ## JUst make everything uniform then...
            
            weights = np.ones(len(data[:,1]))/float(len(data[:,1]))
        else:    
            data_ = (data[:,1]-min)/(diff)
            # data_ = data[:,1]-min
            sum = np.sum(data_, 0) 
            weights = data_ / sum
        self._data = copy.deepcopy(data)
        # print ("Weights: " + str(weights))
        # print ("Data: " + str(self._data))
        self._data[:,1] = np.array(weights, dtype='float64')
        # Delete old samples
        # self._samples = []
        # print ("Done computing pdf data: " + str(self._data))
        
    def getSampleNeighbours(self, sample_):
        
        self._samples
        neighbours = []
        points = copy.deepcopy(self._samples)
        points.sort(key = lambda p: np.sqrt(np.sum((p._data - sample_)**2 )))
        
        return points[1]._data
            
    
    def drawSample(self):
        samp = np.random.choice(self._data[:,0], p=np.array(self._data[:,1], dtype='float64'))
        # samp = np.random.choice(self._data[:,0])
        # print ("Sample: " + str(samp))
        # print ("Sample type: " + str(samp[0].dtype))
        ### Should really make this dependent on the distance to its neighbours...
        if ( self.getSettings()['variance_scalling'] == "adaptive" ):
            neighbour = self.getSampleNeighbours(samp)
            samples = self.generateSamplesFromNormal(samp, 1, variance_=(np.fabs(samp - neighbour)*0.5))
        else:
            samples = self.generateSamplesFromNormal(samp, 1, variance_=self.getSettings()['variance_scalling'])
        return samples[0]
    