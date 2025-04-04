import sys
import math
from actor.ActorInterface import ActorInterface
from model.ModelUtil import reward_smoother

class OpenAIGymActor(ActorInterface):
    
    def __init__(self, discrete_actions, experience):
        super(OpenAIGymActor,self).__init__(discrete_actions, experience)
#         self._target_vel_weight=self._settings["target_velocity_decay"]
#         self._target_vel = self._settings["target_velocity"]
        # self._target_vel = self._settings["target_velocity"]
        self._end_of_episode=False
    
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping=bootstrapping)
        
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, sim, action_, bootstrapping=False):
        import numpy as np
        # Actor should be FIRST here
        if ("use_entropy_reward" in self._settings
            and (
                (self._settings["use_entropy_reward"] == "ICM") or
                (self._settings["use_entropy_reward"] == "ICM_bonus") or
                (self._settings["use_entropy_reward"] == "ICM+SMiRL")
                or (self._settings["use_entropy_reward"] == "RND")
            )):
            state = sim.getState()
            
        ob, reward, done, info  = sim.step(action_)
        if (sim.getMovieWriter() is not None
            and (sim.movieWriterSupport())):
            ### If the sim does not have it's own writing support
            vizData = sim.getEnvironment().getFullViewData()
            image_ = np.zeros((vizData.shape))
            for row in range(len(vizData)):
                image_[row] = vizData[len(vizData)-row - 1]
            ## Convert to int to get rid of warning.
            image_ = np.array(image_, dtype="uint8")
            sim.getMovieWriter().append_data(image_)
        
        self._count = self._count + 1
        if ("use_entropy_reward" in self._settings
            and (self._settings["use_entropy_reward"] == True)):
            self.updateScalling(sim.getState())
            self._reward_sum = self._reward_sum + np.mean(reward)
            reward = self.entropyReward(sim.getState())
        elif ("use_entropy_reward" in self._settings
            and (self._settings["use_entropy_reward"] == "bonus")):
            self.updateScalling(sim.getState())
            bs_w = self._settings["entropy_reward_weight"]
            bs_r = self.entropyReward(sim.getState())
            self._reward_sum = self._reward_sum + np.mean(reward)
            # print ("entropy: ", bs_r, " reward: ", reward)
            reward = (bs_r * bs_w) + reward
            return ob, reward, done, info
        elif ("use_entropy_reward" in self._settings
            and (self._settings["use_entropy_reward"] == "action")):
            action_z = action_[0][-self._settings["encoding_vector_size"]]
            self.updateScalling(sim.getState())
            bs_r = self.entropyReward(sim.getState(), action = action_z)
            self._reward_sum = self._reward_sum + np.mean(reward)
            
            reward = bs_r
            return ob, reward, done, info
        elif ("use_entropy_reward" in self._settings
            and (self._settings["use_entropy_reward"] == "ICM")):
            bs_r = self.rewardICM(state, action_, sim.getState())
            self._reward_sum = self._reward_sum + np.mean(reward)
            
            reward = bs_r
            # print ("ICM reward:", reward)
            return ob, reward, done, info
        
        elif ("use_entropy_reward" in self._settings
            and (self._settings["use_entropy_reward"] == "ICM_bonus")):
            bs_r = self.rewardICM(state, action_, sim.getState())
            self._reward_sum = self._reward_sum + np.mean(reward)
            bs_w = self._settings["entropy_reward_weight"]
            # reward = bs_r
            # print ("ICM reward:", bs_r)
            reward = (bs_r * bs_w) + reward
        elif ("use_entropy_reward" in self._settings
            and (self._settings["use_entropy_reward"] == "ICM+SMiRL")):
            r_icm = self.rewardICM(state, action_, sim.getState())
            self.updateScalling(sim.getState())
            self._reward_sum = self._reward_sum + np.mean(reward)
            r_smirl = self.entropyReward(sim.getState())
            self._reward_sum = self._reward_sum + np.mean(reward)
            bs_w = self._settings["entropy_reward_weight"]
            # reward = bs_r
            # print ("ICM reward:", bs_r)
            reward = (r_icm * bs_w) + r_smirl
        elif ("use_entropy_reward" in self._settings
            and (self._settings["use_entropy_reward"] == "RND")):
            bs_r = self.rewardRND(state, action_, sim.getState())
            self._reward_sum = self._reward_sum + np.mean(reward)
            
            reward = bs_r
            # print ("ICM reward:", reward)
            return ob, reward, done, info
        else:
            self._reward_sum = self._reward_sum + np.mean(reward)        
        return ob, reward, done, info
        
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
        # falls_ = [[not e] for e in exp._fallen]
        if (type(exp._end_of_episode) is list):
            falls_ = [[not e] for e in exp._end_of_episode]
        else:
            falls_ = not exp._end_of_episode
        return falls_
        """
        if ( exp._end_of_episode ):
            return 1
        else:
            return 0
        """