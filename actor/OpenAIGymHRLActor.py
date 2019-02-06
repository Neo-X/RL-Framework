import sys
import math
from actor.OpenAIGymActor import OpenAIGymActor
from model.ModelUtil import reward_smoother

class OpenAIGymHRLActor(OpenAIGymActor):
    
    def __init__(self, discrete_actions, experience):
        super(OpenAIGymHRLActor,self).__init__(discrete_actions, experience)
    
    
    def init(self):
        from util.SimulationUtil import createNetworkModel, createRLAgent
        import numpy as np
        if ('llc_policy_model_path' in self._settings):
            print ("Loading pre compiled network")
            file_name=self._settings['llc_policy_model_path']
            
            import json
            file = open(file_name)
            settings = json.load(file)
            file.close()
            
            settings["load_saved_model"] = True
            # settings["load_saved_model"] = "network_and_scales"
            model = createRLAgent(settings['agent_name'], state_bounds=settings["state_bounds"],
                                   discrete_actions=np.array([[0]]), 
                                   reward_bounds=settings["reward_bounds"], 
                                   settings=settings)
    
                
            self._llc_policy = model
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping=bootstrapping)
        
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        import numpy as np
        # Actor should be FIRST here
        # print ("Action: " + str(action_))
        # dist = exp.getEnvironment().step(action_, bootstrapping=bootstrapping)
        reward_ = 0
        steps = 0
        for i in range(5):
            llc_state_ = exp.getSubPolicyState()
            ### replace target direction/goal with HLC action
            # print ("hlc action: ", action_)
            hlc_action = np.concatenate((action_, np.zeros((llc_state_.shape[0], 1))),axis=1)
            # print ("hlc action: ", hlc_action)
            llc_state_[:,-3:] = hlc_action
            # print ("New llc state: ", llc_state_)
            llc_action = self._llc_policy.predict(llc_state_)
            reward = exp.step(llc_action)
            reward_ = reward_ + reward
            steps = steps + 1
        reward = reward_ / float(steps)
        self._reward_sum = self._reward_sum + np.mean(reward)
        return reward
        
    def updateAction(self, sim, action_):
        import numpy as np
        action_ = np.array(action_, dtype='float64')
        sim.getEnvironment().updateAction(action_)
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        """
            Returns True when the agent is still going (not end of episode)
            return false when the agent has fallen (end of episode)
        """
        if ( exp._end_of_episode ):
            return 1
        else:
            return 0