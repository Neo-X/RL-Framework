
"""
    A class that contains most of the logging and plotting logic
"""

from util.SimulationUtil import getDataDirectory, getAgentNameString, getAgentName, getAgentNameString

class Plotter(object):
    
    def __init__(self, settings):
        import matplotlib
        import copy
        self._settings = copy.deepcopy(settings)
        
        if ((self._settings['visualize_learning'] == False) 
            and (self._settings['save_trainData'] == True) ):
            import matplotlib
            matplotlib.use('Agg')
            print("********Using non interactive matplotlib interface")

        ### It would be nice to move this to its own files/classes as well.
        if ( self._settings['save_trainData'] or self._settings['visualize_learning']):
            from RLVisualize import RLVisualize
            if (self._settings['train_forward_dynamics']
                or self._settings['debug_critic']
                or self._settings['debug_actor']):
                from NNVisualize import NNVisualize
            
        if self._settings['visualize_learning']:
            title = getAgentNameString(self._settings['agent_name'])
            k = title.rfind(".") + 1
            if (k > len(title)): ## name does not contain a .
                k = 0 
            title = str(self._settings['sim_config_file'])
            self._env_name = self._settings['sim_config_file']
            self._rlv = RLVisualize(title=title + " agent on " + str(self._env_name), settings=self._settings)
            self._rlv.setInteractive()
            self._rlv.init()
        if (self._settings['train_forward_dynamics']):
            if self._settings['visualize_learning']:
                title = self._settings['forward_dynamics_model_type']
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                self._nlv = NNVisualize(title=str("Dynamics Model") + " with " + title, settings=self._settings)
                self._nlv.setInteractive()
                self._nlv.init()
            if (self._settings['train_reward_predictor']):
                if self._settings['visualize_learning']:
                    title = self._settings['forward_dynamics_model_type']
                    k = title.rfind(".") + 1
                    if (k > len(title)): ## name does not contain a .
                        k = 0 
                    
                    title = title[k:]
                    self._rewardlv = NNVisualize(title=str("Reward Model") + " with " + title, settings=self._settings)
                    self._rewardlv.setInteractive()
                    self._rewardlv.init()
                 
        if (self._settings['debug_critic']):
            if (self._settings['visualize_learning']):
                title = getAgentNameString(self._settings['agent_name'])
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                self._critic_loss_viz = NNVisualize(title=str("Critic Loss") + " with " + title)
                self._critic_loss_viz.setInteractive()
                self._critic_loss_viz.init()
                self._critic_regularization_viz = NNVisualize(title=str("Critic Reg Cost") + " with " + title)
                self._critic_regularization_viz.setInteractive()
                self._critic_regularization_viz.init()
            
        if (self._settings['debug_actor']):
            if (self._settings['visualize_learning']):
                title = getAgentNameString(self._settings['agent_name'])
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                self._actor_loss_viz = NNVisualize(title=str("Actor Loss") + " with " + title)
                self._actor_loss_viz.setInteractive()
                self._actor_loss_viz.init()
                self._actor_regularization_viz = NNVisualize(title=str("Actor Reg Cost") + " with " + title)
                self._actor_regularization_viz.setInteractive()
                self._actor_regularization_viz.init()
        
        
    def updatePlots(self, masterAgent, trainData):
        ### Lets always save a figure for the learning...
        import numpy as np
        directory= getDataDirectory(self._settings)
        
        # print ("masterAgent.getExperience().samples() >= batch_size: ", masterAgent.getExperience().samples(), " >= ", batch_size)
        error = 0
        rewards = 0
        criticLosses = []
        criticRegularizationCosts = []
        actorLosses = []
        actorRegularizationCosts = []
        if masterAgent.samples() >= batch_size:
            states, actions, result_states, rewards, falls, G_ts, exp_actions, advantage, datas = masterAgent.get_batch(batch_size, 0)
            # print ("Batch size: " + str(batch_size))
            masterAgent.reset()
            error = masterAgent.bellman_error()
            # error = np.mean(np.fabs(error), axis=1)
            # print ("Error: ", error)
            # bellman_errors.append(np.mean(np.fabs(error)))
            bellman_errors.append(error)
            if (self._settings['debug_critic']):
                masterAgent.reset()
                if ((("train_LSTM" in self._settings)
                and (self._settings["train_LSTM"] == True))
                    or (("train_LSTM_Critic" in self._settings)
                    and (self._settings["train_LSTM_Critic"] == True))):
                    batch_size_lstm = 4
                    if ("lstm_batch_size" in self._settings):
                        batch_size_lstm = self._settings["lstm_batch_size"][1]
                    states_, actions_, result_states_, rewards_, falls_, G_ts_, exp_actions, advantage_, datas = masterAgent.getExperience().get_multitask_trajectory_batch(batch_size=min(batch_size_lstm, masterAgent.getExperience().samplesTrajectory()))
                    loss__ = masterAgent.getPolicy().get_critic_loss(states_, actions_, rewards_, result_states_)
                else:
                    loss__ = masterAgent.getPolicy().get_critic_loss(states, actions, rewards, result_states)
                criticLosses.append(loss__)
                regularizationCost__ = masterAgent.getPolicy().get_critic_regularization()
                criticRegularizationCosts.append(regularizationCost__)
                
            
            if not all(np.isfinite(np.mean(error, axis=0))):
                print ("Bellman Error is Nan: " + str(error) + str(np.isfinite(error)))
                # if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                print ("States: " + str(states) + " ResultsStates: " + str(result_states) + " Rewards: " + str(rewards) + " Actions: " + str(actions) + " Falls: ", str(falls))
                sys.exit()
            
            error = np.mean(np.fabs(error), axis=1)
            if np.mean(error) > 10000:
                print ("Error to big: ")
                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                    print (states, actions, rewards, result_states)
                
        if (self._settings['train_forward_dynamics']):
            if ( 'keep_seperate_fd_exp_buffer' in self._settings 
                 and (self._settings['keep_seperate_fd_exp_buffer'])):
                states, actions, result_states, rewards, falls, G_ts, exp_actions, advantage, datas = masterAgent.getFDBatch(batch_size)
            masterAgent.reset()
            if (("train_LSTM_FD" in self._settings)
                and (self._settings["train_LSTM_FD"] == True)):
                batch_size_lstm_fd = 4
                if ("lstm_batch_size" in self._settings):
                    batch_size_lstm_fd = self._settings["lstm_batch_size"][0]
                ### This can consume a lot of memory if trajectories are long...
                state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_, datas = masterAgent.getFDmultitask_trajectory_batch(batch_size=4)
                dynamicsLoss = masterAgent.getForwardDynamics().bellman_error(state_, action_, resultState_, reward_)
            else:
                dynamicsLoss = masterAgent.getForwardDynamics().bellman_error(states, actions, result_states, rewards)
            if (type(dynamicsLoss) == 'list'):
                dynamicsLoss = np.mean([np.mean(np.fabs(dfl)) for dfl in dynamicsLoss])
            else:
                dynamicsLoss = np.mean(np.fabs(dynamicsLoss))
            dynamicsLosses.append(dynamicsLoss)
            if (self._settings['train_reward_predictor']):
                masterAgent.reset()
                if (("train_LSTM_Reward" in self._settings)
                    and (self._settings["train_LSTM_Reward"] == True)):
                    batch_size_lstm_fd = 4
                    if ("lstm_batch_size" in self._settings):
                        batch_size_lstm_fd = self._settings["lstm_batch_size"][0]
                    ### This can consume a lot of memory if trajectories are long...
                    state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_, datas = masterAgent.getFDmultitask_trajectory_batch(batch_size=4)
                    dynamicsRewardLoss = masterAgent.getForwardDynamics().reward_error(state_, action_, resultState_, reward_)
                else:
                    dynamicsRewardLoss = masterAgent.getForwardDynamics().reward_error(states, actions, result_states, rewards)
                
                if (type(dynamicsRewardLoss) == 'list'):
                    dynamicsRewardLoss = np.mean([np.mean(np.fabs(drl)) for drl in dynamicsRewardLoss])
                else:
                    dynamicsRewardLoss = np.mean(np.fabs(dynamicsRewardLoss))

                dynamicsRewardLosses.append(dynamicsRewardLoss)
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                if (self._settings['train_forward_dynamics']):
                    print ("Round: " + str(trainData["round"]) + " of ", rounds,  ", Epoch: " + str(epoch) + " p: " + str(p) + " With mean reward: " + str(np.mean(rewards)) + " bellman error: " + str(error) + " ForwardPredictionLoss: " + str(dynamicsLoss))
                else:
                    print ("Round: " + str(trainData["round"]) + " of ", rounds,  ", Epoch: " + str(epoch) + " p: " + str(p) + " With mean reward: " + str(np.mean(rewards)) + " bellman error: " + str(error))
            # discounted_values.append(discounted_sum)
            
        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
            print ("Master agent experience size: " + str(masterAgent.samples()))
        # print ("**** Master agent experience size: " + str(learning_workers[0]._agent._expBuff.samples()))
        
                
        """
        pr.disable()
        f = open('x.prof', 'a')
        pstats.Stats(pr, stream=f).sort_stats('time').print_stats()
        f.close()
        """
        
        if ( self._settings['save_trainData'] and (not self._settings['visualize_learning'])
             and (self._settings["train_actor"] == True)):
            rlv_ = RLVisualize(title=str(self._settings['sim_config_file']) + " agent on " + str(self._env_name), settings=self._settings)
            rlv_.init()
            rlv_.updateBellmanError(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
            rlv_.updateReward(np.array(trainData["mean_reward"]), np.array(trainData["std_reward"]))
            rlv_.updateDiscountError(np.fabs(trainData["mean_discount_error"]), np.array(trainData["std_discount_error"]))
            rlv_.redraw()
            rlv_.saveVisual(directory+getAgentName())
            rlv_.finish()
            del rlv_
        if self._settings['visualize_learning'] and (self._settings["train_actor"] == True):
            self._rlv.updateBellmanError(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
            self._rlv.updateReward(np.array(trainData["mean_reward"]), np.array(trainData["std_reward"]))
            self._rlv.updateDiscountError(np.fabs(trainData["mean_discount_error"]), np.array(trainData["std_discount_error"]))
            self._rlv.redraw()
            self._rlv.setInteractiveOff()
            self._rlv.saveVisual(directory+getAgentName())
            self._rlv.setInteractive()
        
        if (self._settings['train_forward_dynamics'] and self._settings['save_trainData']
            and (not self._settings['visualize_learning'])):
            nlv_ = NNVisualize(title=str("Dynamics Model") + " with " + str(self._settings['sim_config_file']), settings=self._settings)
            nlv_.init()
            nlv_.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
            nlv_.redraw()
            nlv_.saveVisual(directory+"trainingGraphNN")
            nlv_.finish()
            del nlv_
            if (self._settings['train_reward_predictor']):
                rewardlv_ = NNVisualize(title=str("Reward Model") + " with " + str(self._settings['sim_config_file']), settings=self._settings)
                rewardlv_.init()
                rewardlv_.updateLoss(np.array(trainData["mean_forward_dynamics_reward_loss"]), np.array(trainData["std_forward_dynamics_reward_loss"]))
                rewardlv_.redraw()
                rewardlv_.saveVisual(directory+"rewardTrainingGraph")
                rewardlv_.finish()
                del rewardlv_
        if (self._settings['visualize_learning'] and self._settings['train_forward_dynamics']):
            self._nlv.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
            self._nlv.redraw()
            self._nlv.setInteractiveOff()
            self._nlv.saveVisual(directory+"trainingGraphNN")
            self._nlv.setInteractive()
            if (self._settings['train_reward_predictor']):
                self._rewardlv.updateLoss(np.array(trainData["mean_forward_dynamics_reward_loss"]), np.array(trainData["std_forward_dynamics_reward_loss"]))
                self._rewardlv.redraw()
                self._rewardlv.setInteractiveOff()
                self._rewardlv.saveVisual(directory+"rewardTrainingGraph")
                self._rewardlv.setInteractive()
        if (self._settings['debug_critic']):
            
            masterAgent.reset()
            if ((("train_LSTM" in self._settings)
            and (self._settings["train_LSTM"] == True))
                or (("train_LSTM_Critic" in self._settings)
                and (self._settings["train_LSTM_Critic"] == True))):
                batch_size_lstm = 4
                if ("lstm_batch_size" in self._settings):
                    batch_size_lstm = self._settings["lstm_batch_size"][1]
                states_, actions_, result_states_, rewards_, falls_, G_ts_, exp_actions, advantage_, datas = masterAgent.getExperience().get_multitask_trajectory_batch(batch_size=min(batch_size_lstm, masterAgent.getExperience().samplesTrajectory()))
                loss__ = masterAgent.getPolicy().get_critic_loss(states_, actions_, rewards_, result_states_)
            else:
                loss__ = masterAgent.getPolicy().get_critic_loss(states, actions, rewards, result_states)
            criticLosses.append(loss__)
            regularizationCost__ = masterAgent.getPolicy().get_critic_regularization()
            criticRegularizationCosts.append(regularizationCost__)
                        
            mean_criticLosses = np.mean([np.mean(cl) for cl in criticLosses])
            std_criticLosses = np.mean([np.std(acl) for acl in criticLosses])
            logExperimentData(trainData, "mean_critic_loss", mean_criticLosses, self._settings)
            logExperimentData(trainData, "std_critic_loss", std_criticLosses, self._settings)
            if (self._settings['visualize_learning']):
                self._critic_loss_viz.updateLoss(np.array(trainData["mean_critic_loss"]), np.array(trainData["std_critic_loss"]))
                self._critic_loss_viz.redraw()
                self._critic_loss_viz.setInteractiveOff()
                self._critic_loss_viz.saveVisual(directory+"criticLossGraph")
                self._critic_loss_viz.setInteractive()
            
            mean_criticRegularizationCosts = np.mean(criticRegularizationCosts)
            std_criticRegularizationCosts = np.std(criticRegularizationCosts)
            logExperimentData(trainData, "mean_critic_regularization_cost", mean_criticRegularizationCosts, self._settings)
            logExperimentData(trainData, "std_critic_regularization_cost", std_criticRegularizationCosts, self._settings)
            if (self._settings['visualize_learning']):
                self._critic_regularization_viz.updateLoss(np.array(trainData["mean_critic_regularization_cost"]), np.array(trainData["std_critic_regularization_cost"]))
                self._critic_regularization_viz.redraw()
                self._critic_regularization_viz.setInteractiveOff()
                self._critic_regularization_viz.saveVisual(directory+"criticRegularizationGraph")
                self._critic_regularization_viz.setInteractive()
            
        if (self._settings['debug_actor']):
            
            masterAgent.reset()
            loss__ = [p_.getPolicy().get_actor_loss(states, actions, rewards, result_states, advantage) for p_ in masterAgent.getAgents() ]
            actorLosses.append(np.mean(loss__))
            regularizationCost__ = [p_.getPolicy().get_actor_regularization() for p_ in masterAgent.getAgents() ]
            actorRegularizationCosts.append(np.mean(regularizationCost__))
            
            mean_actorLosses = np.mean([np.mean(acL) for acL in actorLosses])
            std_actorLosses = np.mean([np.std(acl) for acl in actorLosses])
            logExperimentData(trainData, "mean_actor_loss", mean_actorLosses, settings)
            logExperimentData(trainData, "std_actor_loss", std_actorLosses, settings)
            
            
            if (self._settings['visualize_learning']):
                self._actor_loss_viz.updateLoss(np.array(trainData["mean_actor_loss"]), np.array(trainData["std_actor_loss"]))
                self._actor_loss_viz.redraw()
                self._actor_loss_viz.setInteractiveOff()
                self._actor_loss_viz.saveVisual(directory+"actorLossGraph")
                self._actor_loss_viz.setInteractive()
            
            mean_actorRegularizationCosts = np.mean(actorRegularizationCosts)
            std_actorRegularizationCosts = np.std(actorRegularizationCosts)
            logExperimentData(trainData, "mean_actor_regularization_cost", mean_actorRegularizationCosts, self._settings)
            logExperimentData(trainData, "std_actor_regularization_cost", std_actorRegularizationCosts, self._settings)
            actorRegularizationCosts = []
            if (self._settings['visualize_learning']):
                self._actor_regularization_viz.updateLoss(np.array(trainData["mean_actor_regularization_cost"]), np.array(trainData["std_actor_regularization_cost"]))
                self._actor_regularization_viz.redraw()
                self._actor_regularization_viz.setInteractiveOff()
                self._actor_regularization_viz.saveVisual(directory+"actorRegularizationGraph")
                self._actor_regularization_viz.setInteractive()
        
    def finish(self):
        print("Delete any plots being used")
    
        if self._settings['visualize_learning']:    
            self._rlv.finish()
        if (self._settings['train_forward_dynamics']):
            if self._settings['visualize_learning']:
                self._nlv.finish()
            if (self._settings['train_reward_predictor']):
                if self._settings['visualize_learning']:
                    self._rewardlv.finish()
                 
        if (self._settings['debug_critic']):
            if (self._settings['visualize_learning']):
                self._critic_loss_viz.finish()
                self._critic_regularization_viz.finish()
        if (self._settings['debug_actor']):
            if (self._settings['visualize_learning']):
                self._actor_loss_viz.finish()
                self._actor_regularization_viz.finish()
            