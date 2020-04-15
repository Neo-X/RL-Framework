
"""
    A class that contains most of the logging and plotting logic
"""


class Plotter(object):
    
    def __init__(self, settings):
        import matplotlib
        
        self._settings = settings
        
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
            if (self._settings['environment_type'] == "open_AI_Gym"):
                self._settings['environment_type'] = self._settings['sim_config_file']
            rlv = RLVisualize(title=title + " agent on " + str(self._settings['environment_type']), settings=self._settings)
            rlv.setInteractive()
            rlv.init()
        if (self._settings['train_forward_dynamics']):
            if self._settings['visualize_learning']:
                title = self._settings['forward_dynamics_model_type']
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                nlv = NNVisualize(title=str("Dynamics Model") + " with " + title, settings=self._settings)
                nlv.setInteractive()
                nlv.init()
            if (self._settings['train_reward_predictor']):
                if self._settings['visualize_learning']:
                    title = self._settings['forward_dynamics_model_type']
                    k = title.rfind(".") + 1
                    if (k > len(title)): ## name does not contain a .
                        k = 0 
                    
                    title = title[k:]
                    rewardlv = NNVisualize(title=str("Reward Model") + " with " + title, settings=self._settings)
                    rewardlv.setInteractive()
                    rewardlv.init()
                 
        if (self._settings['debug_critic']):
            criticLosses = []
            criticRegularizationCosts = [] 
            if (self._settings['visualize_learning']):
                title = getAgentNameString(self._settings['agent_name'])
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                critic_loss_viz = NNVisualize(title=str("Critic Loss") + " with " + title)
                critic_loss_viz.setInteractive()
                critic_loss_viz.init()
                critic_regularization_viz = NNVisualize(title=str("Critic Reg Cost") + " with " + title)
                critic_regularization_viz.setInteractive()
                critic_regularization_viz.init()
            
        if (self._settings['debug_actor']):
            actorLosses = []
            actorRegularizationCosts = []            
            if (self._settings['visualize_learning']):
                title = getAgentNameString(self._settings['agent_name'])
                k = title.rfind(".") + 1
                if (k > len(title)): ## name does not contain a .
                    k = 0 
                title = title[k:]
                actor_loss_viz = NNVisualize(title=str("Actor Loss") + " with " + title)
                actor_loss_viz.setInteractive()
                actor_loss_viz.init()
                actor_regularization_viz = NNVisualize(title=str("Actor Reg Cost") + " with " + title)
                actor_regularization_viz.setInteractive()
                actor_regularization_viz.init()
        
        
    def updatePlots(self, masterAgent):
        ### Lets always save a figure for the learning...
        if ( self._settings['save_trainData'] and (not self._settings['visualize_learning'])
             and (self._settings["train_actor"] == True)):
            rlv_ = RLVisualize(title=str(self._settings['sim_config_file']) + " agent on " + str(self._settings['environment_type']), self._settings=self._settings)
            rlv_.init()
            rlv_.updateBellmanError(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
            rlv_.updateReward(np.array(trainData["mean_reward"]), np.array(trainData["std_reward"]))
            rlv_.updateDiscountError(np.fabs(trainData["mean_discount_error"]), np.array(trainData["std_discount_error"]))
            rlv_.redraw()
            rlv_.saveVisual(directory+getAgentName())
            rlv_.finish()
            del rlv_
        if self._settings['visualize_learning'] and (self._settings["train_actor"] == True):
            rlv.updateBellmanError(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
            rlv.updateReward(np.array(trainData["mean_reward"]), np.array(trainData["std_reward"]))
            rlv.updateDiscountError(np.fabs(trainData["mean_discount_error"]), np.array(trainData["std_discount_error"]))
            rlv.redraw()
            rlv.setInteractiveOff()
            rlv.saveVisual(directory+getAgentName())
            rlv.setInteractive()
        
        if (self._settings['train_forward_dynamics'] and self._settings['save_trainData']
            and (not self._settings['visualize_learning'])):
            nlv_ = NNVisualize(title=str("Dynamics Model") + " with " + str(self._settings['sim_config_file']), self._settings=self._settings)
            nlv_.init()
            nlv_.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
            nlv_.redraw()
            nlv_.saveVisual(directory+"trainingGraphNN")
            nlv_.finish()
            del nlv_
            if (self._settings['train_reward_predictor']):
                rewardlv_ = NNVisualize(title=str("Reward Model") + " with " + str(self._settings['sim_config_file']), self._settings=self._settings)
                rewardlv_.init()
                rewardlv_.updateLoss(np.array(trainData["mean_forward_dynamics_reward_loss"]), np.array(trainData["std_forward_dynamics_reward_loss"]))
                rewardlv_.redraw()
                rewardlv_.saveVisual(directory+"rewardTrainingGraph")
                rewardlv_.finish()
                del rewardlv_
        if (self._settings['visualize_learning'] and self._settings['train_forward_dynamics']):
            nlv.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
            nlv.redraw()
            nlv.setInteractiveOff()
            nlv.saveVisual(directory+"trainingGraphNN")
            nlv.setInteractive()
            if (self._settings['train_reward_predictor']):
                rewardlv.updateLoss(np.array(trainData["mean_forward_dynamics_reward_loss"]), np.array(trainData["std_forward_dynamics_reward_loss"]))
                rewardlv.redraw()
                rewardlv.setInteractiveOff()
                rewardlv.saveVisual(directory+"rewardTrainingGraph")
                rewardlv.setInteractive()
        if (self._settings['debug_critic']):
            
            mean_criticLosses = np.mean([np.mean(cl) for cl in criticLosses])
            std_criticLosses = np.mean([np.std(acl) for acl in criticLosses])
            logExperimentData(trainData, "mean_critic_loss", mean_criticLosses, self._settings)
            logExperimentData(trainData, "std_critic_loss", std_criticLosses, self._settings)
            criticLosses = []
            if (self._settings['visualize_learning']):
                critic_loss_viz.updateLoss(np.array(trainData["mean_critic_loss"]), np.array(trainData["std_critic_loss"]))
                critic_loss_viz.redraw()
                critic_loss_viz.setInteractiveOff()
                critic_loss_viz.saveVisual(directory+"criticLossGraph")
                critic_loss_viz.setInteractive()
            
            mean_criticRegularizationCosts = np.mean(criticRegularizationCosts)
            std_criticRegularizationCosts = np.std(criticRegularizationCosts)
            logExperimentData(trainData, "mean_critic_regularization_cost", mean_criticRegularizationCosts, self._settings)
            logExperimentData(trainData, "std_critic_regularization_cost", std_criticRegularizationCosts, self._settings)
            criticRegularizationCosts = []
            if (self._settings['visualize_learning']):
                critic_regularization_viz.updateLoss(np.array(trainData["mean_critic_regularization_cost"]), np.array(trainData["std_critic_regularization_cost"]))
                critic_regularization_viz.redraw()
                critic_regularization_viz.setInteractiveOff()
                critic_regularization_viz.saveVisual(directory+"criticRegularizationGraph")
                critic_regularization_viz.setInteractive()
            
        if (self._settings['debug_actor']):
            
            mean_actorLosses = np.mean([np.mean(acL) for acL in actorLosses])
            std_actorLosses = np.mean([np.std(acl) for acl in actorLosses])
            logExperimentData(trainData, "mean_actor_loss", mean_actorLosses, self._settings)
            logExperimentData(trainData, "std_actor_loss", std_actorLosses, self._settings)
            actorLosses = []
            if (self._settings['visualize_learning']):
                actor_loss_viz.updateLoss(np.array(trainData["mean_actor_loss"]), np.array(trainData["std_actor_loss"]))
                actor_loss_viz.redraw()
                actor_loss_viz.setInteractiveOff()
                actor_loss_viz.saveVisual(directory+"actorLossGraph")
                actor_loss_viz.setInteractive()
            
            mean_actorRegularizationCosts = np.mean(actorRegularizationCosts)
            std_actorRegularizationCosts = np.std(actorRegularizationCosts)
            logExperimentData(trainData, "mean_actor_regularization_cost", mean_actorRegularizationCosts, self._settings)
            logExperimentData(trainData, "std_actor_regularization_cost", std_actorRegularizationCosts, self._settings)
            actorRegularizationCosts = []
            if (self._settings['visualize_learning']):
                actor_regularization_viz.updateLoss(np.array(trainData["mean_actor_regularization_cost"]), np.array(trainData["std_actor_regularization_cost"]))
                actor_regularization_viz.redraw()
                actor_regularization_viz.setInteractiveOff()
                actor_regularization_viz.saveVisual(directory+"actorRegularizationGraph")
                actor_regularization_viz.setInteractive()
        
    def finish(self):
        print("Delete any plots being used")
    
        if self._settings['visualize_learning']:    
            rlv.finish()
        if (self._settings['train_forward_dynamics']):
            if self._settings['visualize_learning']:
                nlv.finish()
            if (self._settings['train_reward_predictor']):
                if self._settings['visualize_learning']:
                    rewardlv.finish()
                 
        if (self._settings['debug_critic']):
            if (self._settings['visualize_learning']):
                critic_loss_viz.finish()
                critic_regularization_viz.finish()
        if (self._settings['debug_actor']):
            if (self._settings['visualize_learning']):
                actor_loss_viz.finish()
                actor_regularization_viz.finish()
            