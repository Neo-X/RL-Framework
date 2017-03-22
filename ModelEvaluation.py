

# import cPickle
import dill
import sys
import gc
# from theano.compile.io import Out
sys.setrecursionlimit(50000)
# from sim.PendulumEnvState import PendulumEnvState
# from sim.PendulumEnv import PendulumEnv
from multiprocessing import Process, Queue
# from pathos.multiprocessing import Pool
import threading
import time
import copy
import numpy as np
from model.ModelUtil import *
# import memory_profiler
# import resource

# class SimWorker(threading.Thread):
class SimWorker(Process):
    
    def __init__(self, namespace, input_queue, output_queue, actor, exp, model, discount_factor, action_space_continuous, 
                 settings, print_data, p, validation, eval_episode_data_queue):
        super(SimWorker, self).__init__()
        self._input_queue= input_queue
        self._output_queue = output_queue
        self._eval_episode_data_queue = eval_episode_data_queue
        self._actor = actor
        self._exp = exp
        self._model = model
        self._discount_factor = discount_factor
        self._action_space_continuous= action_space_continuous
        self._settings= settings
        self._print_data=print_data
        self._p= p
        self._validation=validation
        self._max_iterations = settings['rounds'] + settings['epochs'] * 32
        self._iteration = 0
        self._namespace = namespace # A way to pass messages between processes
    
    def current_mem_usage(self):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.

    def setEnvironment(self, exp_):
        """
            Set the environment instance to use
        """
        self._exp = exp_
        self._model.setEnvironment(self._exp)
        
    # @profile(precision=5)
    def run(self):
        # from pympler import summary
        # from pympler import muppy
        
        # print ("SW model: ", self._model.getPolicy())
        print ("Thread: ", self._model._exp)
        ## This is no needed if there is one thread only...
        if (int(self._settings["num_available_threads"]) > 1): 
            from util.SimulationUtil import createEnvironment
            self._exp = createEnvironment(str(self._settings["sim_config_file"]), self._settings['environment_type'], self._settings)
            self._exp.getActor().init()   
            self._exp.getEnvironment().init()
            ## The sampler might need this new model is threads > 1
            self._model.setEnvironment(self._exp)
        
        print ('Worker started')
        # do some initialization here
        while True:
            eval=False
            episodeData = self._input_queue.get()
            if episodeData == None:
                break
            if episodeData['type'] == "eval":
                eval=True
                episodeData = episodeData['data']
                "Sim worker evaluating episode"
            else:
                episodeData = episodeData['data']
            if (self._model.getPolicy() == None): # cheap hack for now
                self._model.setPolicy(copy.deepcopy(self._namespace.model))
            if ( (self._settings["train_forward_dynamics"]) and ( self._model.getForwardDynamics() == None ) ):
                self._model.setForwardDynamics(copy.deepcopy(self._namespace.forwardDynamicsModel))
            # print('\tWorker maximum memory usage: %.2f (mb)' % (self.current_mem_usage()))
            # print ("Nums samples in worker: ", self._namespace.experience.samples())
            p = self._namespace.p
            print ("Sim worker Size of state input Queue: " + str(self._input_queue.qsize()))
            if p < 0.1:
                p = 0.1
            self._p = p
            # print ("sim worker p: " + str(self._p))
            if (eval): ## No action exploration
                out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                        anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                        print_data=self._print_data, p=0.0, validation=True, evaluation=eval)
            else:    
                out = self.simEpochParallel(actor=self._actor, exp=self._exp, model=self._model, discount_factor=self._discount_factor, 
                        anchors=episodeData, action_space_continuous=self._action_space_continuous, settings=self._settings, 
                        print_data=self._print_data, p=self._p, validation=self._validation, evaluation=eval)
            self._iteration += 1
            # if self._p <= 0.0:
            
            #    self._output_queue.put(out)
            (tuples, discounted_sum, q_value, evalData) = out
            (states, actions, rewards, result_states, falls) = tuples
            ## Hack for now just update after ever episode
            if (eval):
                self._eval_episode_data_queue.put(out)
            else:
                print("Updating sim policies:")
                
                self._model.getPolicy().setNetworkParameters(copy.deepcopy(self._namespace.agentPoly))
                if (self._settings['train_forward_dynamics']):
                    self._model.getForwardDynamics().setNetworkParameters(copy.deepcopy(self._namespace.forwardNN))
                # gc.collect()
                
            # print ("Actions: " + str(actions))
            # all_objects = muppy.get_objects()
            # sum1 = summary.summarize(all_objects)
            # summary.print_(sum1)
        print ("Simulation Worker Complete: ")
        self._exp.finish()
        
    def simEpochParallel(self, actor, exp, model, discount_factor, anchors=None, action_space_continuous=False, settings=None, print_data=False, p=0.0, validation=False, epoch=0, evaluation=False):
        out = simEpoch(actor, exp, model, discount_factor, anchors=anchors, action_space_continuous=action_space_continuous, settings=settings, 
                       print_data=print_data, p=p, validation=validation, epoch=epoch, evaluation=evaluation, _output_queue=self._output_queue )
        return out
    
    

def simEpoch(actor, exp, model, discount_factor, anchors=None, action_space_continuous=False, settings=None, print_data=False, 
             p=0.0, validation=False, epoch=0, evaluation=False, _output_queue=None, bootstraping=False, visualizeEvaluation=None):
    """
        
        evaluation: If Ture than the simulation is being evaluated and the episodes will not terminate early.
        bootstraping: is used to collect initial random actions for the state bounds to be calculated and to init the expBuffer
        epoch: is an integer that can be used to help create repeatable episodes to evaluation
        _output_queue: is the queue exp tuples should be put in so the learning agents can pull them out
        p:  is the probability of selecting a random action
        actor: 
    """
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
        omega = settings["omega"]
    
    action_selection = range(len(settings["discrete_actions"]))   
    reward_bounds = np.array(settings['reward_bounds'] )
    
    pa=None
    epsilon = settings["epsilon"]
    # Actor should be FIRST here
    exp.getActor().initEpoch()
    if validation:
        exp.generateValidation(anchors, epoch)
    else:
        exp.generateEnvironmentSample()
        
    exp.getEnvironment().initEpoch()
    actor.init()
    state_ = exp.getState()
    # pa = model.predict(state_)
    if (not bootstraping):
        q_values_ = [model.q_value(state_)]
    else:
        q_values_ = []
    viz_q_values_ = []
    # q_value = model.q_value(state_)
    # print ("Updated parameters: " + str(model._pol.getNetworkParameters()[3]))
    # print ("q_values_: " + str(q_value) + " Action: " + str(action_))
    # original_val = q_value
    discounted_sum = 0;
    discounted_sums = [];
    state_num=0
    i_=0
    reward_=0
    states = [] 
    actions = []
    rewards = []
    falls = []
    result_states = []
    evalDatas=[]
    
    # while not exp.getEnvironment().endOfEpoch():
    for i_ in range(settings['max_epoch_length']):
        
        # if (exp.getEnvironment().endOfEpoch() or (reward_ < settings['reward_lower_bound'])):
        if (exp.getEnvironment().endOfEpoch() or ((reward_ < settings['reward_lower_bound']) 
                                                  and
                                                  (not evaluation))):
            evalDatas.append(actor.getEvaluationData()/float(settings['max_epoch_length']))
            discounted_sums.append(discounted_sum)
            discounted_sum=0
            state_num=0
            exp.getActor().initEpoch()
            if validation:
                exp.generateValidation(anchors, (epoch * settings['max_epoch_length']) + i_)
            else:
                exp.generateEnvironmentSample()
                
            exp.getEnvironment().initEpoch()
            actor.init()
            state_ = exp.getState()
            if (not bootstraping):
                q_values_.append(model.q_value(state_))
        # state = exp.getEnvironment().getState()
        state_ = exp.getState()
        if (not (visualizeEvaluation == None)):
            viz_q_values_.append(model.q_value(state_)[0])
            if (len(viz_q_values_)>30):
                 viz_q_values_.pop(0)
            # print ("viz_q_values_: ", viz_q_values_ )
            # print ("np.zeros(len(viz_q_values_)): ", np.zeros(len(viz_q_values_)))
            visualizeEvaluation.updateLoss(viz_q_values_, np.zeros(len(viz_q_values_)))
            visualizeEvaluation.redraw()
            # visualizeEvaluation.setInteractiveOff()
            # visualizeEvaluation.saveVisual(directory+"criticLossGraph")
            # visualizeEvaluation.setInteractive()
        # print ("Initial State: " + str(state_))
        # print ("State: " + str(state.getParams()))
        # val_act = exp.getActor().getModel().maxExpectedActionForState(state)
        # action_ = model.predict(state_)
            # print ("Get best action: ")
        # pa = model.predict(state_)
        action=None
        if action_space_continuous:
            """
                epsilon greedy action select
                pa1 is best action from policy
                ra1 is the noisy policy action action
                ra2 is the random action
                e is proabilty to select random action
                0 <= e < omega < 1.0
            """
            r = np.random.rand(1)[0]
            if r < (epsilon * p): # explore random actions
                
                r2 = np.random.rand(1)[0]
                if (r2 < (omega * p)) or bootstraping:# explore hand crafted actions
                    # return ra2
                    # randomAction = randomUniformExporation(action_bounds) # Completely random action
                    # action = randomAction
                    action = np.random.choice(action_selection)
                    action__ = actor.getActionParams(action)
                    action = action__
                    # print ("Discrete action choice: ", action, " epsilon * p: ", epsilon * p)
                else : # add noise to current policy
                    # return ra1
                    pa = model.predict(state_)
                    if (settings['exploration_method'] == 'gaussian_random'):
                        # action = randomExporation(settings["exploration_rate"], pa)
                        action = randomExporation(settings["exploration_rate"], pa, action_bounds)
                    elif ((settings['exploration_method'] == 'thompson')):
                        # print ('Using Thompson sampling')
                        action = thompsonExploration(model, settings["exploration_rate"], state_)
                    else:
                        print ("Exploration method unknown: " + str(settings['exploration_method']))
                        sys.exit(1)
                    # randomAction = randomUniformExporation(action_bounds) # Completely random action
                    # randomAction = random.choice(action_selection)
                    if (settings["use_model_based_action_optimization"] and (np.random.rand(1)[0] < settings["model_based_action_omega"])):
                        # Need to be using a forward dynamics deep network for this
                        action = getOptimalAction(model.getForwardDynamics(), model.getPolicy(), state_)
                    # print ("Exploration: Before action: ", pa, " after action: ", action, " epsilon: ", epsilon * p )
            else: # exploit policy
                # return pa1
                ## For sampling method to skip sampling during evaluation.
                pa = model.predict(state_, evaluation_=evaluation)
                
                action = pa
                # print ("Exploitation: ", action , " epsilon: ", epsilon * p)
            outside_bounds=False
            action_=None
            if (settings["clamp_actions_to_stay_inside_bounds"] or (settings['penalize_actions_outside_bounds'])):
                (action_, outside_bounds) = clampActionWarn(action, action_bounds)
                if (settings['clamp_actions_to_stay_inside_bounds']):
                    action = action_
            if (settings["visualize_forward_dynamics"]):
                predicted_next_state = model.getForwardDynamics().predict(np.array(state_), action)
                # exp.visualizeNextState(state_, action) # visualize current state
                exp.visualizeNextState(predicted_next_state, action)
                
                action = model.predict(state_)
                actions = []
                dirs = []
                deltas = np.linspace(-0.5,0.5,10)
                for d in range(len(deltas)):
                    action_ = np.zeros_like(action)
                    for i in range(len(action_)):
                        action_[i] = action[i]
                    action_[0] = action[0] + deltas[d] 
                    action_new_ = getOptimalAction2(model.getForwardDynamics(), model.getPolicy(), action_, state_)
                    # actions.append(action_new_)
                    actions.append(action_)
                    if ( (action_new_[0] - action_[0]) > 0 ):
                        dirs.append(1.0)
                    else:
                        dirs.append(-1.0)
                    
                # return _getOptimalAction(forwardDynamicsModel, model, action, state)
                
                # action_ = _getOptimalAction(model.getForwardDynamics(), model.getPolicy(), action, state_)
                exp.getEnvironment().visualizeActions(actions, dirs)
                ## The perfect action?
                exp.getEnvironment().visualizeAction([2.25])
                
            
            if (not settings["train_actor"]): # hack to use debug critic only
                """
                    action = np.random.choice(action_selection)
                    action__ = actor.getActionParams(action)
                    action = action__
                    
                    pa = model.predict(state_)
                    action = pa
                """
                pass
                # action=[0.2]
            reward_ = actor.actContinuous(exp,action)
            agent_not_fell = actor.hasNotFallen(exp)
            if (outside_bounds and settings['penalize_actions_outside_bounds']):
                reward_ = reward_ + reward_bounds[0] # TODO: this penalty should really be a function of the distance the action was outside the bounds 
            # print ("Action: ", action, " reward: ", reward_, " p: ", p)
        elif not action_space_continuous:
            """
            action = random.choice(action_selection)
            action = eGreedy(pa, action, epsilon * p)
            reward_ = actor.act(exp, action)
            """
            pa = model.predict(state_)
            action = random.choice(action_selection)
            action = eGreedy(pa, action, epsilon * p)
            # print("Action selection:", action_selection, " action: ", action)
            action__ = actor.getActionParams(action)
            action = [action]
            # print ("Action selected: " + str(action__))
            # reward = act(action)
            # print ("performing action: ", action)
            reward_ = actor.actContinuous(exp, action__, bootstrapping=True)
            agent_not_fell = actor.hasNotFallen(exp)
            # print ("performed action: ", reward)
        """
        if (agent_not_fell == 0):
            print ("Agent fell ", agent_not_fell, " with reward: ", reward_, " from action: ", action)
            # reward_=0
        """  
        if ((reward_ >= settings['reward_lower_bound'] )):
            # discounted_sum = discounted_sum + (((math.pow(discount_factor,state_num) * reward_))) # *(1.0-discount_factor))
            discounted_sum = discounted_sum + (((math.pow(discount_factor,state_num) * (reward_ * (1.0-discount_factor) )))) # *(1.0-discount_factor))
        # print ("discounted_sum: ", discounted_sum)
        resultState = exp.getState()
        # print ("Result State: " + str(resultState))
        # _val_act = exp.getActor().getModel().maxExpectedActionForState(resultState)
        # bellman_error.append(val_act[0] - (reward + _val_act[0]))
        # For testing remove later
        if (settings["use_back_on_track_forcing"] and (not evaluation)):
            exp.getControllerBackOnTrack()
            
        # print ("Value: ", model.q_value(state_), " Action " + str(pa) + " Reward: " + str(reward_) )
        if print_data:
            # print ("State " + str(state_) + " action " + str(pa) + " newState " + str(resultState) + " Reward: " + str(reward_))
            print ("Value: ", model.q_value(state_), " Action " + str(pa) + " Reward: " + str(reward_) ) 
            pass     
            # print ("Python Reward: " + str(reward(state_, resultState)))
            
        if ( (reward_ >= settings['reward_lower_bound'] ) or evaluation):
            states.extend(state_)
            actions.append(action)
            rewards.append([reward_])
            result_states.append(resultState)
            falls.append([agent_not_fell])
            # print ("falls: ", falls)
            # values.append(value)
            if (_output_queue != None and ((not evaluation) or (not bootstraping))): # for multi-threading
                # _output_queue.put((norm_state(state_, model.getStateBounds()), [norm_action(action, model.getActionBounds())], [reward_], norm_state(state_, model.getStateBounds()))) # TODO: Should these be scaled?
                _output_queue.put((state_, action, [reward_], resultState, [agent_not_fell]))
            state_num += 1
        else:
            print ("****Reward was to bad: ", reward_)
        pa = None
        i_ += 1
        
    evalDatas.append(actor.getEvaluationData()/float(settings['max_epoch_length']))
    evalData = [np.mean(evalDatas)]
    discounted_sums.append(discounted_sum)
    discounted_sum = np.mean(discounted_sums)
    q_value = np.mean(q_values_)
    if print_data:
        print ("Evaluation: ", str(evalData)) 
    # print ("Evaluation Data: ", evalData)
        # print ("Current Tuple: " + str(experience.current()))
    tuples = (states, actions, rewards, result_states, falls)
    return (tuples, discounted_sum, q_value, evalData)
    

# @profile(precision=5)
def evalModel(actor, exp, model, discount_factor, anchors=None, action_space_continuous=False, settings=None, print_data=False, p=0.0, evaluation=False, visualizeEvaluation=None):
    print ("Evaluating model:")
    j=0
    discounted_values = []
    bellman_errors = []
    reward_over_epocs = []
    values = []
    evalDatas = []
    epoch_=0
    for i in range(anchors): # half the anchors
        (tuples, discounted_sum, value, evalData) = simEpoch(actor, exp, 
                model, discount_factor, anchors=i, action_space_continuous=action_space_continuous, 
                settings=settings, print_data=print_data, p=0.0, validation=True, epoch=epoch_, evaluation=evaluation,
                visualizeEvaluation=visualizeEvaluation)
        epoch_ = epoch_ + 1
        (states, actions, rewards, result_states, falls) = tuples
        # print (states, actions, rewards, result_states, discounted_sum, value)
        # print ("Evaluated Actions: ", actions)
        # print ("Evaluated Rewards: ", rewards)
        if model.getExperience().samples() > settings['batch_size']:
            _states, _actions, _result_states, _rewards, falls = model.getExperience().get_batch(settings['batch_size'])
            error = model.bellman_error(_states, _actions, _rewards, _result_states, falls)
        else :
            error = [[0]]
            print ("Error: not enough samples")
        # states, actions, result_states, rewards = experience.get_batch(64)
        # error = model.bellman_error(states, actions, rewards, result_states)
        # print (states, actions, rewards, result_states, discounted_sum, value)
        error = np.mean(np.fabs(error))
        # print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " With reward_sum: " + str(np.sum(rewards)) + " bellman error: " + str(error))
        discounted_values.append(discounted_sum)
        values.append(value)
        # print ("Rewards over eval epoch: ", rewards)
        # This works better because epochs can terminate early, which is bad.
        reward_over_epocs.append(np.mean(np.array(rewards)))
        bellman_errors.append(error)
        evalDatas.append(evalData)
        
    print ("Reward for min epoch: " + str(np.argmax(reward_over_epocs)) + " is " + str(np.max(reward_over_epocs)))
    print ("reward_over_epocs" + str(reward_over_epocs))
    print ("Discounted sum: ", discounted_values)
    print ("Initial values: ", values)
    mean_reward = np.mean(reward_over_epocs)
    std_reward = np.std(reward_over_epocs)
    mean_bellman_error = np.mean(bellman_errors)
    std_bellman_error = np.std(bellman_errors)
    mean_discount_error = np.mean(np.array(discounted_values) - np.array(values))
    std_discount_error = np.std(np.array(discounted_values) - np.array(values))
    mean_eval = np.mean(evalDatas)
    std_eval = np.std(evalDatas)
    
    discounted_values = []
    reward_over_epocs = []
    bellman_errors = []
        
    return (mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error,
            mean_eval, std_eval)

# @profile(precision=5)
def evalModelParrallel(input_anchor_queue, eval_episode_data_queue, model, settings, anchors=None):
    print ("Evaluating model Parrallel:")
    j=0
    discounted_values = []
    bellman_errors = []
    reward_over_epocs = []
    values = []
    evalDatas = []
    epoch_=0
    i = 0 
    while i < anchors: # half the anchors
        
        j = 0
        while (j < settings['num_available_threads']) and ( (i + j) < anchors):
            episodeData = {}
            episodeData['data'] = i
            episodeData['type'] = 'eval'
            input_anchor_queue.put(episodeData)
            j += 1
            
        # for anchs in anchors: # half the anchors
        j = 0
        while (j < settings['num_available_threads']) and ( (i + j) < anchors):
            (tuples, discounted_sum, value, evalData) =  eval_episode_data_queue.get()
            j += 1
            """
            simEpoch(actor, exp, 
                    model, discount_factor, anchors=anchs, action_space_continuous=action_space_continuous, 
                    settings=settings, print_data=print_data, p=0.0, validation=True, epoch=epoch_, evaluation=evaluation,
                    visualizeEvaluation=visualizeEvaluation)
            """
            epoch_ = epoch_ + 1
            (states, actions, rewards, result_states, falls) = tuples
            # print (states, actions, rewards, result_states, discounted_sum, value)
            # print ("Evaluated Actions: ", actions)
            # print ("Evaluated Rewards: ", rewards)
            if model.getExperience().samples() > settings['batch_size']:
                _states, _actions, _result_states, _rewards, falls = model.getExperience().get_batch(settings['batch_size'])
                error = model.bellman_error(_states, _actions, _rewards, _result_states, falls)
            else :
                error = [[0]]
                print ("Error: not enough samples")
            # states, actions, result_states, rewards = experience.get_batch(64)
            # error = model.bellman_error(states, actions, rewards, result_states)
            # print (states, actions, rewards, result_states, discounted_sum, value)
            error = np.mean(np.fabs(error))
            # print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " With reward_sum: " + str(np.sum(rewards)) + " bellman error: " + str(error))
            discounted_values.append(discounted_sum)
            values.append(value)
            # print ("Rewards over eval epoch: ", rewards)
            # This works better because epochs can terminate early, which is bad.
            reward_over_epocs.append(np.mean(np.array(rewards)))
            bellman_errors.append(error)
            evalDatas.append(evalData)
        i += j
        
    print ("Reward for min epoch: " + str(np.argmax(reward_over_epocs)) + " is " + str(np.max(reward_over_epocs)))
    print ("reward_over_epocs" + str(reward_over_epocs))
    print ("Discounted sum: ", discounted_values)
    print ("Initial values: ", values)
    mean_reward = np.mean(reward_over_epocs)
    std_reward = np.std(reward_over_epocs)
    mean_bellman_error = np.mean(bellman_errors)
    std_bellman_error = np.std(bellman_errors)
    mean_discount_error = np.mean(np.array(discounted_values) - np.array(values))
    std_discount_error = np.std(np.array(discounted_values) - np.array(values))
    mean_eval = np.mean(evalDatas)
    std_eval = np.std(evalDatas)
    
    discounted_values = []
    reward_over_epocs = []
    bellman_errors = []
        
    return (mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error,
            mean_eval, std_eval)
# @profile(precision=5)
def collectExperience(actor, exp_val, model, settings):
    from util.ExperienceMemory import ExperienceMemory
    action_selection = range(len(settings["discrete_actions"]))
    print ("Action selection: " + str(action_selection))
    # state_bounds = np.array(settings['state_bounds'])
    # state_bounds = np.array([[0],[0]])
    reward_bounds=np.array(settings["reward_bounds"])
    action_bounds = np.array(settings["action_bounds"], dtype=float)
    state_bounds = np.array(settings['state_bounds'], dtype=float)
    
    if (settings["bootsrap_with_discrete_policy"]) and (settings['bootsrap_samples'] > 0):
        (states, actions, resultStates, rewards_, falls_) = collectExperienceActionsContinuous(actor, exp_val, model, settings['bootsrap_samples'], settings=settings, action_selection=action_selection)
        # states = np.array(states)
        # states = np.append(states, state_bounds,0) # Adding that already specified bounds will ensure the final calculated is beyond these
        print (" Shape states: ", states.shape)
        state_bounds = np.ones((2,states.shape[1]))
        
        state_avg = states[:settings['bootsrap_samples']].mean(0)
        state_stddev = states[:settings['bootsrap_samples']].std(0)*2
        reward_avg = rewards_[:settings['bootsrap_samples']].mean(0)
        reward_stddev = rewards_[:settings['bootsrap_samples']].std(0)*2
        action_avg = actions[:settings['bootsrap_samples']].mean(0)
        action_stddev = actions[:settings['bootsrap_samples']].std(0)*2
        if (settings['state_normalization'] == "minmax"):
            state_bounds[0] = states[:settings['bootsrap_samples']].min(0)
            state_bounds[1] = states[:settings['bootsrap_samples']].max(0)
            # reward_bounds[0] = rewards_[:settings['bootsrap_samples']].min(0)
            # reward_bounds[1] = rewards_[:settings['bootsrap_samples']].max(0)
            # action_bounds[0] = actions[:settings['bootsrap_samples']].min(0)
            # action_bounds[1] = actions[:settings['bootsrap_samples']].max(0)
        elif (settings['state_normalization'] == "variance"):
            state_bounds[0] = state_avg - state_stddev
            state_bounds[1] = state_avg + state_stddev
            # reward_bounds[0] = reward_avg - reward_stddev
            # reward_bounds[1] = reward_avg + reward_stddev
            # action_bounds[0] = action_avg - action_stddev
            # action_bounds[1] = action_avg + action_stddev
        elif (settings['state_normalization'] == "given"):
            # pass # Use bound specified in file
            state_bounds = np.array(settings['state_bounds'], dtype=float)
        else:
            print ("State scaling strategy unknown: ", (settings['state_normalization']))
            
        ## Cast data to the proper type
        state_bounds = np.array(state_bounds, dtype=settings['float_type'])
        reward_bounds = np.array(reward_bounds, dtype=settings['float_type'])
        action_bounds = np.array(action_bounds, dtype=settings['float_type'])
            
        if settings['action_space_continuous']:
            experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings = settings)
        else:
            experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
        experience.setSettings(settings)
        
        
        print ("State Mean:" + str(state_avg))
        print ("State Variance: " + str(state_stddev))
        print ("Reward Mean:" + str(reward_avg))
        print ("Reward Variance: " + str(reward_stddev))
        print ("Action Mean:" + str(action_avg))
        print ("Action Variance: " + str(action_stddev))
        print ("Max State:" + str(state_bounds[1]))
        print ("Min State:" + str(state_bounds[0]))
        print ("Max Reward:" + str(reward_bounds[1]))
        print ("Min Reward:" + str(reward_bounds[0]))
        print ("Max Action:" + str(action_bounds[1]))
        print ("Min Action:" + str(action_bounds[0]))
        
        experience.setStateBounds(state_bounds)
        experience.setRewardBounds(reward_bounds)
        experience.setActionBounds(action_bounds)
        
        for state, action, resultState, reward_, fall_ in zip(states, actions, resultStates, rewards_, falls_):
            if reward_ > settings['reward_lower_bound']: # Skip is reward gets too bad, skips nan too?u
                if settings['action_space_continuous']:
                    # experience.insert(norm_state(state, state_bounds), norm_action(action, action_bounds), norm_state(resultState, state_bounds), norm_reward([reward_], reward_bounds))
                    experience.insert(state, action, resultState, [reward_], [fall_])
                else:
                    experience.insert(state, [action], resultState, [reward_], [falls_])
            else:
                print ("Tuple with reward: " + str(reward_) + " skipped")
        # sys.exit()
    else: ## Most like performing continuation learning
        if settings['action_space_continuous']:
            experience = ExperienceMemory(len(model.getStateBounds()[0]), len(model.getActionBounds()[0]), settings['expereince_length'], continuous_actions=True, settings = settings)
        else:
            experience = ExperienceMemory(len(model.getStateBounds()[0]), 1, settings['expereince_length'])
            experience.setSettings(settings)
            experience.setStateBounds(model.getStateBounds())
            experience.setRewardBounds(model.getRewardBounds())
            experience.setActionBounds(model.getActionBounds())
        """
        (states, actions, resultStates, rewards_) = collectExperienceActionsContinuous(exp, settings['expereince_length'], settings=settings, action_selection=action_selection)
        # states = np.array(states)
        state_bounds[0] = states.min(0)
        state_bounds[1] = states.max(0)
        reward_bounds[0][0] = rewards_.min(0)
        print ("Max State:" + str(state_bounds[1]))
        print ("Min State:" + str(state_bounds[0]))
        print ("Min Reward:" + str(reward_bounds[0]))
        """
        
        
    return  experience, state_bounds, reward_bounds, action_bounds

# @profile(precision=5)
def collectExperienceActionsContinuous(actor, exp, model, samples, settings, action_selection):
    i = 0
    states = []
    actions = []
    resultStates = []
    rewards = []
    falls = []
    # anchor_data_file = open(settings["anchor_file"])
    # _anchors = getAnchors(anchor_data_file)
    # print ("Length of anchors epochs: " + str(len(_anchors)))
    # anchor_data_file.close()
    while i < samples:
        # Actor should be FIRST here
        out = simEpoch(actor=actor, exp=exp, model=model, discount_factor=settings['discount_factor'], anchors=i, 
                               action_space_continuous=settings['action_space_continuous'], settings=settings, print_data=False,
                                p=100.0, validation=settings['train_on_validation_set'], bootstraping=True)
        # if self._p <= 0.0:
        #    self._output_queue.put(out)
        (tuples, discounted_sum_, q_value_, evalData) = out
        (states_, actions_, rewards_, result_states_, falls_) = tuples
        print ("Shape other states_: ", np.array(states_).shape)
        print ("Shape other action_: ", np.array(actions_).shape)
        states.extend(states_)
        actions.extend(actions_)
        rewards.extend(rewards_)
        resultStates.extend(result_states_)
        falls.extend(falls_)
        
        i=i+len(states_)
        print("Number of Experience samples so far: ", i)
        # print ("States: ", states)
        # print ("Actions: ", actions)
        # print ("Rewards: ", rewards)
        # print ("ResultStates: ", resultStates)
        

    print ("Done collecting experience.")
    return (np.array(states), np.array(actions), np.array(resultStates), np.array(rewards), np.array(falls_))  


def modelEvaluation(settings_file_name):
    
    from model.ModelUtil import getSettings
    settings = getSettings(settings_file_name)
    settings['shouldRender'] = True
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
    
    ## Theano needs to be imported after the flags are set.
    # from ModelEvaluation import *
    # from model.ModelUtil import *
    from ModelEvaluation import SimWorker, evalModelParrallel, collectExperience
    # from model.ModelUtil import validBounds
    from model.LearningAgent import LearningAgent, LearningWorker
    from util.SimulationUtil import validateSettings, createEnvironment, createRLAgent, createActor
    from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel
    
    
    from util.ExperienceMemory import ExperienceMemory
    from RLVisualize import RLVisualize
    from NNVisualize import NNVisualize
    
    # from model.ModelUtil import *
    # from actor.ActorInterface import *
    # from util.SimulationUtil import *
    
    # anchor_data_file = open(settings["anchor_file"])
    # _anchors = getAnchors(anchor_data_file)
    # anchor_data_file.close()
    model_type= settings["model_type"]
    directory= getDataDirectory(settings)
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    # num_states=settings["num_states"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    # max_reward=settings["max_reward"]
    batch_size=settings["batch_size"]
    state_bounds = np.array(settings['state_bounds'])
    action_space_continuous=settings["action_space_continuous"]  
    discrete_actions = np.array(settings['discrete_actions'])
    num_actions= discrete_actions.shape[0]
    reward_bounds=np.array(settings["reward_bounds"])
    action_space_continuous=settings['action_space_continuous']
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
    
    print ("Sim config file name: " + str(settings["sim_config_file"]))
    
    ### Using a wrapper for the type of actor now
    if action_space_continuous:
        experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
    else:
        experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
    # actor = ActorInterface(discrete_actions)
    actor = createActor(str(settings['environment_type']),settings, experience)
    masterAgent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"_Best.pkl"
    # file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+".pkl"
    f = open(file_name, 'r')
    model = dill.load(f)
    f.close()
    print ("State Length: ", len(model.getStateBounds()[0]) )
    
    if (settings['train_forward_dynamics']):
        file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+"_Best.pkl"
        # file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+".pkl"
        f = open(file_name_dynamics, 'r')
        forwardDynamicsModel = dill.load(f)
        f.close()
    
    if ( settings["use_transfer_task_network"] ):
        task_directory = getTaskDataDirectory(settings)
        file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"_Best.pkl"
        f = open(file_name, 'r')
        taskModel = dill.load(f)
        f.close()
        # copy the task part from taskModel to model
        print ("Transferring task portion of model.")
        model.setTaskNetworkParameters(taskModel)

    # this is the process that selects which game to play
    
    exp = createEnvironment(str(settings["sim_config_file"]), str(settings['environment_type']), settings)

    if (settings['train_forward_dynamics']):
        # actor.setForwardDynamicsModel(forwardDynamicsModel)
        forwardDynamicsModel.setActor(actor)
        masterAgent.setForwardDynamics(forwardDynamicsModel)
        # forwardDynamicsModel.setEnvironment(exp)
    # actor.setPolicy(model)
    
    exp.getActor().init()   
    exp.getEnvironment().init()
    expected_value_viz=None
    if (settings['visualize_expected_value']):
        expected_value_viz = NNVisualize(title=str("Expected Value") + " with " + str(settings["model_type"]), settings=settings)
        expected_value_viz.setInteractive()
        expected_value_viz.init()
        criticLosses = []
        
    masterAgent.setSettings(settings)
    masterAgent.setExperience(experience)
    masterAgent.setPolicy(model)
    
    
    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModel(actor, exp, masterAgent, discount_factor, anchors=settings['eval_epochs'], 
                                                                                                                        action_space_continuous=action_space_continuous, settings=settings, print_data=True, evaluation=True,
                                                                                                                        visualizeEvaluation=expected_value_viz)
        # simEpoch(exp, model, discount_factor=discount_factor, anchors=_anchors[:settings['eval_epochs']][9], action_space_continuous=True, settings=settings, print_data=True, p=0.0, validation=True)
    
    """
    workers = []
    input_anchor_queue = Queue(settings['queue_size_limit'])
    output_experience_queue = Queue(settings['queue_size_limit'])
    for process in range(settings['num_available_threads']):
         # this is the process that selects which game to play
        exp = characterSim.Experiment(c)
        if settings['environment_type'] == 'pendulum_env_state':
            print ("Using Environment Type: " + str(settings['environment_type']))
            exp = PendulumEnvState(exp)
        elif settings['environment_type'] == 'pendulum_env':
            print ("Using Environment Type: " + str(settings['environment_type']))
            exp = PendulumEnv(exp)
        else:
            print ("Invalid environment type: " + str(settings['environment_type']))
            sys.exit()
                
        
        exp.getActor().init()   
        exp.getEnvironment().init()
        
        w = SimWorker(input_anchor_queue, output_experience_queue, exp, model, discount_factor, action_space_continuous=action_space_continuous, 
                settings=settings, print_data=False, p=0.0, validation=True)
        w.start()
        workers.append(w)
        
    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error = evalModelParrallel(
        input_anchor_queue, output_experience_queue, discount_factor, anchors=_anchors[:settings['eval_epochs']], action_space_continuous=action_space_continuous, settings=settings)
    
    for w in workers:
        input_anchor_queue.put(None)
       """ 
    print ("Average Reward: " + str(mean_reward))
    
    
if __name__ == "__main__":
    
    modelEvaluation(sys.argv[1])
