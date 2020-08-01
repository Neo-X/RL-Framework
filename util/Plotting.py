
"""
    A class that contains most of the logging and plotting logic
"""

from util.SimulationUtil import getDataDirectory, getAgentNameString, getAgentName, getAgentNameString
from RLVisualize import RLVisualize
from NNVisualize import NNVisualize

def display_paths_gif(paths, logdir, fps=10, max_outputs=8, counter=0):
    import moviepy.editor as mpy
    import numpy as np
    images = []
    for i in range(len(paths)):
        images_ = paths[i]['rendering']
#         images_ = images_[:max_outputs]
    #     images = np.clip(images, 0.0, 1.0)
    #     images = (images * 255.0).astype(np.uint8)
        images.append(images_)
    images = np.concatenate(images, axis=-3) ## concatenate the images into one row.
    clip = mpy.ImageSequenceClip(list(images), fps=fps)
        # clip.write_videofile(logdir+str(global_counter)+".mp4", fps=fps)
    
#     clip.write_gif(logdir+"all"+".gif", fps=fps)
    clip.write_videofile(logdir+"all"+".mp4", fps=fps)
#     clip.write_videofile(logdir+"all"+".webm", fps=fps)
    
def display_gif(paths, logdir, fps=10, max_outputs=8, counter=0):
    import moviepy.editor as mpy
    import numpy as np
    images = []
    for i in range(len(paths)):
        images_ = paths[i]['rendering']
#         images_ = images_[:max_outputs]
    #     images = np.clip(images, 0.0, 1.0)
    #     images = (images * 255.0).astype(np.uint8)
        images.append(images_)
    images = images[:max_outputs]
    images = np.concatenate(images, axis=-2)
    clip = mpy.ImageSequenceClip(list(images), fps=fps)
    # clip.write_videofile(logdir+str(global_counter)+".mp4", fps=fps)
    
    #     os.makedirs(video_dir, exist_ok = True)
#     clip.write_gif(logdir+str(counter)+".gif", fps=fps)
#     clip.write_videofile(logdir+str(counter)+".webm", fps=fps)
    clip.write_videofile(logdir+str(counter)+".mp4", fps=fps)
    
def create_filmStrip(paths, logdir, fps=10, max_outputs=8, counter=0):
   """
   Create a filmstrip of the video
   """
   pass

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
                
        # paramSampler = exp_val.getActor().getParamSampler()
        self._best_eval =-100000000.0
#         self._mean_eval = best_eval * 10
        self._best_dynamicsLosses = self._best_eval*-1.0
#         self._mean_dynamicsLosses = best_dynamicsLosses * 10 
                
                
    def getSettings(self):
        return self._settings
        
        
    def updatePlots(self, masterAgent, trainData, sampler, out, p, settings):
        ### Lets always save a figure for the learning...
        from util.SimulationUtil import createEnvironment, logExperimentData, saveData, logExperimentImage
        from util.utils import current_mem_usage
        self._settings = settings
        
        
        (tuples, discounted_sum, q_value, evalData) = out
        (__states, __actions, __result_states, __rewards, __falls, __G_ts, advantage__, exp_actions__, datas__) = tuples
        import numpy as np
        directory= getDataDirectory(self._settings)
        if ( 'value_function_batch_size' in self._settings): batch_size=self._settings["value_function_batch_size"]
        else: batch_size=self._settings["batch_size"]
        # print ("masterAgent.getExperience().samples() >= batch_size: ", masterAgent.getExperience().samples(), " >= ", batch_size)
        error = 0
        rewards = 0
        criticLosses = []
        criticRegularizationCosts = []
        actorLosses = []
        actorRegularizationCosts = []
        bellman_errors = []
        dynamicsLosses = []
        dynamicsRewardLosses = []
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
                    
        if (self.getSettings()['debug_actor'] and self.getSettings()['train_actor']):
            
            masterAgent.reset()
            loss__ = [p_.getPolicy().get_actor_loss(states, actions, rewards, result_states, advantage) for p_ in masterAgent.getAgents() ]
            actorLosses.append(np.mean(loss__))
            regularizationCost__ = [p_.getPolicy().get_actor_regularization() for p_ in masterAgent.getAgents() ]
            actorRegularizationCosts.append(np.mean(regularizationCost__))
            
            mean_actorLosses = np.mean([np.mean(acL) for acL in actorLosses])
            std_actorLosses = np.mean([np.std(acl) for acl in actorLosses])
            logExperimentData(trainData, "mean_actor_loss", mean_actorLosses, self._settings)
            logExperimentData(trainData, "std_actor_loss", std_actorLosses, self._settings)
            
            
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
                    batch_size_lstm_fd = 8
#                     if ("lstm_batch_size" in self._settings):
#                         batch_size_lstm_fd = self._settings["lstm_batch_size"][0]
                    ### This can consume a lot of memory if trajectories are long...
                    state_, action_, resultState_, reward_, fall_, G_ts_, exp_actions, advantage_, datas = masterAgent.getFDExperience()[0].get_multitask_trajectory_batch(batch_size=batch_size_lstm_fd, 
                                                                                                                                                                        randomStart = False, randomLength = False)
                    dynamicsRewardLoss = masterAgent.getForwardDynamics().reward_error(state_, action_, resultState_, reward_)
                    
                    if ("compute_model_metrics" in self._settings 
                        and (self._settings["compute_model_metrics"])):
                        modelMetrics = masterAgent.getForwardDynamics().compute_model_metrics(state_, action_)
#                         print ("modelMetrics: ", modelMetrics)
#                         print ("datas: ", datas)
                        for d in range(len(datas)):
                            datas[d].update(modelMetrics[d])
                            for key in datas[d]:
                                ### Put all info data in the logs
                                ### Plot the batch data.
                                nlv_ = NNVisualize(title=key+str(d), settings=self._settings)
                                nlv_.init()
                                data_ = np.array(datas[d][key], dtype="float32").flatten()
                                nlv_.updateLoss(data_, np.zeros_like(data_))
                                nlv_.redraw()
                                nlv_.saveVisual(directory+key+str(d))
                                nlv_.finish()
#                                 (trainData, key+str(d), np.mean([met[key] for met in otherMetrics]), self._settings)
                    if ("log_model_gen_seq_output" in self._settings):
                        from model.ModelUtil import scale_state
                        data = masterAgent.getForwardDynamics().predict_seq(state_, resultState_)
                        (distance_r_weighted, 
                         distance_fd2_weighted, 
                         decode_a, 
                         decode_b,
                         decode_a_vae,
                         decode_b_vae) = data
                        ## Need to reshape the data dn cut out the image data only
                        def fix_shape(data):
                            return np.moveaxis(np.repeat(np.reshape(np.array(scale_state(data, masterAgent.getFDExperience()[0].getStateBounds())[:,:self._settings["fd_num_terrain_features"]] * 255, dtype='uint8'), (len(data),) + (1, 48, 48)), 3, axis=1), 1, -1)
                        display_gif(paths=[{'rendering': fix_shape(traj)} for traj in state_], logdir=directory, fps=30, max_outputs=32, counter="state_")
                        logExperimentImage(path=directory+"state_.mp4", overwrite=True, image_format="mp4", settings=self._settings)
                        display_gif(paths=[{'rendering': fix_shape(traj)} for traj in resultState_], logdir=directory, fps=30, max_outputs=32, counter="resultState_")
                        logExperimentImage(path=directory+"resultState_.mp4", overwrite=True, image_format="mp4", settings=self._settings)
                        display_gif(paths=[{'rendering': fix_shape(traj)} for traj in decode_a], logdir=directory, fps=30, max_outputs=32, counter="decode_a")
                        logExperimentImage(path=directory+"decode_a.mp4", overwrite=True, image_format="mp4", settings=self._settings)
                        display_gif(paths=[{'rendering': fix_shape(traj)} for traj in decode_b], logdir=directory, fps=30, max_outputs=32, counter="decode_b")
                        logExperimentImage(path=directory+"decode_b.mp4", overwrite=True, image_format="mp4", settings=self._settings)
                        display_gif(paths=[{'rendering': fix_shape(traj)} for traj in decode_a_vae], logdir=directory, fps=30, max_outputs=32, counter="decode_a_vae")
                        logExperimentImage(path=directory+"decode_a_vae.mp4", overwrite=True, image_format="mp4", settings=self._settings)
                        display_gif(paths=[{'rendering': fix_shape(traj)} for traj in decode_b_vae], logdir=directory, fps=30, max_outputs=32, counter="decode_b_vae")
                        logExperimentImage(path=directory+"decode_b_vae.mp4", overwrite=True, image_format="mp4", settings=self._settings)
                        
                        reward__r_1 = masterAgent.getForwardDynamics().predict_reward_(state_, state_)
                        logExperimentData(trainData, "reward_self_agreement_agent", np.mean(reward__r_1), self._settings)
                        reward__fd_1 = masterAgent.getForwardDynamics().predict_reward_fd(state_, state_)
                        logExperimentData(trainData, "reward_self_agreement_agent_fd", np.mean(reward__fd_1), self._settings)
                        reward__r_1 = masterAgent.getForwardDynamics().predict_reward_(resultState_, resultState_)
                        logExperimentData(trainData, "reward_self_agreement_expert", np.mean(reward__r_1), self._settings)
                        reward__fd_1 = masterAgent.getForwardDynamics().predict_reward_fd(resultState_, resultState_)
                        logExperimentData(trainData, "reward_self_agreement_expert_fd", np.mean(reward__fd_1), self._settings)
                else:
                    dynamicsRewardLoss = masterAgent.getForwardDnamics().reward_error(states, actions, result_states, rewards)
                
                if (type(dynamicsRewardLoss) == 'list'):
                    dynamicsRewardLoss = np.mean([np.mean(np.fabs(drl)) for drl in dynamicsRewardLoss])
                else:
                    dynamicsRewardLoss = np.mean(np.fabs(dynamicsRewardLoss))

                dynamicsRewardLosses.append(dynamicsRewardLoss)
            # discounted_values.append(discounted_sum)
            
        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
            print ("Master agent experience size: " + str(masterAgent.samples()))
        # print ("**** Master agent experience size: " + str(learning_workers[0]._agent._expBuff.samples()))
        
        if (trainData["round"] % self._settings['plotting_update_freq_num_rounds']) == 0:
            # Running less often helps speed learning up.
            # else:
            if ("skip_rollouts" in self._settings and 
                    (self._settings["skip_rollouts"] == True)):
                mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval, otherMetrics = 0,0,0,0,0,0,0,0, [{}]
                mean_reward = [0] * self._settings["perform_multiagent_training"]
                reward_over_epocs = [[0]] * self._settings["perform_multiagent_training"]
            else:
                rewards__=[]
                reward_over_epocs = []
                for tr in range(len(__rewards)):
                    rewards__ = []
                    for agent_ in range(len(masterAgent.getAgents())): 
                        rewards__.append(np.array(__rewards[tr]).flatten()[agent_::len(masterAgent.getAgents())])
                        # discounted_sum__.append(np.array(discounted_sum).flatten()[agent_::len(masterAgent.getAgents())])
                        # value__.append(np.array(q_value).flatten()[agent_::len(masterAgent.getAgents())])
                        # discount_error__.append(discounted_sum__[agent_] - value__[agent_])
                    rewards_ = [np.mean(rew) for rew in rewards__]
                    # print ("rewards__", tr ,": ", rewards_)
                    reward_over_epocs.append(rewards_)
                    
                if ( ( self._settings["eval_epochs"] == "stochastic")):
                    mean_reward = np.mean(reward_over_epocs)
                    std_reward = np.std(reward_over_epocs)
                    discounted_sum__=[]
                    value__=[]
                    discount_error__ = []
                    mean_bellman_error = 0
                    std_bellman_error = 0
                    mean_discount_error = 0
                    std_discount_error = 0
                    mean_eval = np.mean(reward_over_epocs)
                    std_eval = np.std(reward_over_epocs)
                else:
                    mean_reward, std_reward, mean_bellman_error, std_bellman_error, \
                    mean_discount_error, std_discount_error, mean_eval, std_eval, \
                    otherMetrics = sampler.obtainSamples( masterAgent, rollouts=self._settings['eval_epochs'], p=0, eval=True)

            print ("round_, p, mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error")
            print (trainData["round"], p, mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error)
            if np.mean(mean_bellman_error) > 10000:
                print ("Error to big: ")
            else:
                if (self._settings['train_forward_dynamics']):
                    mean_dynamicsLosses = np.mean(dynamicsLosses)
                    std_dynamicsLosses = np.std(dynamicsLosses)
                    dynamicsLosses = []
                    if (self._settings['train_reward_predictor']):
                        mean_dynamicsRewardLosses = np.mean(dynamicsRewardLosses)
                        std_dynamicsRewardLosses = np.std(dynamicsRewardLosses)
                        dynamicsRewardLosses = []
                    
                    
#                     logExperimentData(trainData, "falls", np.mean([met["falls"] for met in otherMetrics]), self._settings)
                for key in otherMetrics[0].keys() - ['rendering']:
                    ### Put all info data in the logs
                    # print ("attempting to log metrics: ", key, " values: ", [met[key] for met in otherMetrics])
                    
                    logExperimentData(trainData, key, np.mean([met[key] for met in otherMetrics]), self._settings)
                    # pass
#                     logExperimentData(trainData, "mem_usage_sim", np.mean([met["mem_usage_sim"] for met in otherMetrics]), self._settings)

                if ("save_eval_video" in settings and 
                    (settings["save_eval_video"] == True)):
                    display_gif(paths=otherMetrics, logdir=directory, fps=20, max_outputs=32, counter=0)
                    logExperimentImage(path=directory+str(0)+".mp4", overwrite=True, image_format="mp4", settings=self._settings)
                logExperimentData(trainData, "mem_usage_train", np.mean(current_mem_usage()), self._settings)
                logExperimentData(trainData, "mean_reward", mean_reward, self._settings)
                # print ("__rewards: " , reward_over_epocs)
                logExperimentData(trainData, "mean_reward_train", np.mean(reward_over_epocs), self._settings)
                for ag in range(self._settings["perform_multiagent_training"]):
                    logExperimentData(trainData, "mean_reward_agent_"+str(ag), np.mean(mean_reward[ag]), self._settings)
                    mean_train_reward = np.mean(np.array(reward_over_epocs)[:,ag])
                    # print ("mean_train_reward: ", mean_train_reward)
                    logExperimentData(trainData, "mean_reward_train_"+str(ag), mean_train_reward, self._settings)
                logExperimentData(trainData, "std_reward", std_reward, self._settings)
                logExperimentData(trainData, "anneal_p", p, self._settings)
                if (self._settings["train_actor"] == True):
                        
                    logExperimentData(trainData, "mean_bellman_error", np.array([np.mean(er_) for er_ in np.fabs(bellman_errors[0])]), self._settings)
                    logExperimentData(trainData, "std_bellman_error", np.array([np.std(er_) for er_ in bellman_errors[0]]), self._settings)
                    bellman_errors=[]
                    logExperimentData(trainData, "mean_discount_error", mean_discount_error, self._settings)
                    logExperimentData(trainData, "std_discount_error", std_discount_error, self._settings)
                    logExperimentData(trainData, "mean_eval", mean_eval, self._settings)
                    logExperimentData(trainData, "std_eval", std_eval, self._settings)
                # error = np.mean(np.fabs(error), axis=1)
                # trainData["std_bellman_error"].append(std_bellman_error)
                if (self._settings['train_forward_dynamics']):
                    logExperimentData(trainData, "mean_forward_dynamics_loss", mean_dynamicsLosses, self._settings)
                    logExperimentData(trainData, "std_forward_dynamics_loss", std_dynamicsLosses, self._settings)
                    if (self._settings['train_reward_predictor']):
                        logExperimentData(trainData, "mean_forward_dynamics_reward_loss", mean_dynamicsRewardLosses, self._settings)
                        logExperimentData(trainData, "std_forward_dynamics_reward_loss", std_dynamicsRewardLosses, self._settings)
                        
                        
            ## Visulaize some stuff if you want to
            if (int(self._settings["num_available_threads"]) == -1 
                # or (int(self._settings["num_available_threads"]) == 1)
                ): # This is okay if there is one thread only...
                exp_val.updateViz(actor, masterAgent, directory, p=p)
                
        """
        pr.disable()
        f = open('x.prof', 'a')
        pstats.Stats(pr, stream=f).sort_stats('time').print_stats()
        f.close()
        """
        
        if ( self._settings['save_trainData'] and (not self._settings['visualize_learning'])
             and (self._settings["train_actor"] == True)):
            rlv_ = RLVisualize(title=str(self._settings['sim_config_file']), settings=self._settings)
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
        if (self._settings['debug_critic'] and self.getSettings()['train_critic']):
            
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
            
                
        if (trainData["round"] % self.getSettings()['saving_update_freq_num_rounds']) == 0:
        
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['hyper_train']):
                print ("Saving current masterAgent")
            masterAgent.saveTo(directory)
            
            if ( self.getSettings()['train_forward_dynamics'] and 
                 (mean_dynamicsLosses < self._best_dynamicsLosses)):
                self._best_dynamicsLosses = mean_dynamicsLosses
                if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['hyper_train']):
                    print ("Saving BEST current forward dynamics agent: " + str(self._best_dynamicsLosses))
                masterAgent.saveTo(directory, bestFD=True)
                    
            if (mean_eval > self._best_eval):
                self._best_eval = mean_eval
                if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['hyper_train']):
                    print ("Saving BEST current agent: " + str(self._best_eval))
                masterAgent.saveTo(directory, bestPolicy=True)
                
            fp = open(directory+"trainingData_" + str(getAgentNameString(self.getSettings()['agent_name'])) + ".json", 'w')
            # print ("Train data: ", trainData)
            from util.utils import NumpyEncoder 
            import json
            # print ("trainData: ", trainData)
            json.dump(trainData, fp, cls=NumpyEncoder)
            fp.close()
            # draw data

        if "checkpoint_vid_rounds" in self.getSettings() and self.getSettings()["checkpoint_vid_rounds"] is not None \
        and trainData["round"] % self.getSettings()["checkpoint_vid_rounds"] == 0:
           loggingWorkerQueue.put(('checkpoint_vid_rounds', trainData["round"]))


        
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
            