import sys
import math
from actor.ActorInterface import ActorInterface

class TerrainRLActor(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(TerrainRLActor,self).__init__(settings_, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping)
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, sim, action_, bootstrapping=False):
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        # reward = exp.getEnvironment().act(action_)
        sim.act(walk)
        updates_=0
        while (not sim.endOfAction() and (updates_ < 20)):
            sim.update()
            updates_+=1
        reward_ = sim.calcReward()   
        # print ("averageSpeed: ", averageSpeed)
        self._reward_sum = self._reward_sum + reward
        print ("Reward: ", reward_)

        return reward
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        if ( sim.agentHasFallen() ) :
            return 0
        else:
            return 1
        # return not exp.getEnvironment().agentHasFallen()