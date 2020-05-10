

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
# import resources

    
def modelEvaluationParallel(settings_file_name):
    
    from model.ModelUtil import getSettings
    import multiprocessing
    
    settings = getSettings(settings_file_name)
    # settings['shouldRender'] = True
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
    discrete_actions = settings['discrete_actions']
    reward_bounds=np.array(settings["reward_bounds"])
    action_space_continuous=settings['action_space_continuous']
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
        
    input_anchor_queue = multiprocessing.Queue(settings['queue_size_limit'])
    output_experience_queue = multiprocessing.Queue(settings['queue_size_limit'])
    eval_episode_data_queue = multiprocessing.Queue(settings['num_available_threads'])
    mgr = multiprocessing.Manager()
    namespace = mgr.Namespace()
    namespace.p=0
    
    exp_val = None
    
    print ("Sim config file name: " + str(settings["sim_config_file"]))
    
    ### Using a wrapper for the type of actor now
    if action_space_continuous:
        experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['experience_length'], continuous_actions=True, settings=settings)
    else:
        experience = ExperienceMemory(len(state_bounds[0]), 1, settings['experience_length'])
    # actor = ActorInterface(discrete_actions)
    actor = createActor(str(settings['environment_type']),settings, experience)
    masterAgent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    # file_name=directory+getAgentName()+"_Best.pkl"
    # file_name=directory+getAgentName()+".pkl"
    # f = open(file_name, 'r')
    # model = dill.load(f)
    # f.close()
    print ("State Length: ", len(model.getStateBounds()[0]) )
    
    if ( settings["use_transfer_task_network"] ):
        task_directory = getTaskDataDirectory(settings)
        file_name=directory+getAgentName()+"_Best.pkl"
        f = open(file_name, 'r')
        taskModel = dill.load(f)
        f.close()
        # copy the task part from taskModel to model
        print ("Transferring task portion of model.")
        model.setTaskNetworkParameters(taskModel)

    # this is the process that selects which game to play
    
    sim_workers = []
    for process in range(settings['num_available_threads']):
        # this is the process that selects which game to play
        exp_=None
        
        if (int(settings["num_available_threads"]) == 1): # This is okay if there is one thread only...
            print ("Assigning same EXP")
            exp_ = exp_val # This should not work properly for many simulations running at the same time. It could try and evalModel a simulation while it is still running samples 
        print ("original exp: ", exp_)
        if ( settings['use_simulation_sampling'] ):
            
            sampler = createSampler(settings, exp_)
            ## This should be some kind of copy of the simulator not a network
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp_)
            sampler.setForwardDynamics(forwardDynamicsModel)
            # sampler.setPolicy(model)
            agent = sampler
            print ("thread together exp: ", agent._exp)
            # sys.exit()
        else:
            agent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
        
        agent.setSettings(settings)
        w = SimWorker(namespace, input_anchor_queue, output_experience_queue, actor, exp_, agent, discount_factor, action_space_continuous=action_space_continuous, 
                settings=settings, print_data=False, p=0.0, validation=True, eval_episode_data_queue=eval_episode_data_queue, process_random_seed=settings['random_seed']+process )
        # w.start()
        # w._settings['shouldRender']=True
        sim_workers.append(w)
        
    if (int(settings["num_available_threads"]) != 1): # This is okay if there is one thread only...
        for sw in sim_workers:
            print ("Sim worker")
            print (sw)
            sw.start()
            
    ## This needs to be done after the simulation work processes are created
    exp_val = createEnvironment(str(settings["sim_config_file"]), settings['environment_type'], settings, render=settings['shouldRender'])
    # exp_val = createEnvironment(str(settings["sim_config_file"]), settings['environment_type'], settings, render=False)
    exp_val.setActor(actor)
    exp_val.getActor().init()
    exp_val.init()
    
    # exp = createEnvironment(str(settings["sim_config_file"]), str(settings['environment_type']), settings, render=True)
    # exp = createEnvironment(str(settings["sim_config_file"]), str(settings['environment_type']), settings, render=False)
    exp = exp_val
    exp.setActor(actor)
    if (settings['train_forward_dynamics']):
        # actor.setForwardDynamicsModel(forwardDynamicsModel)
        forwardDynamicsModel.setActor(actor)
        masterAgent.setForwardDynamics(forwardDynamicsModel)
        # forwardDynamicsModel.setEnvironment(exp)
    # actor.setPolicy(model)
    
    exp.getActor().init()   
    exp.init()
    if (int(settings["num_available_threads"]) == -1): # This is okay if there is one thread only...
        sim_workers[0].setEnvironment(exp_val)
        sim_workers[0].start()
        
    expected_value_viz=None
    if (settings['visualize_expected_value']):
        expected_value_viz = NNVisualize(title=str("Expected Value") + " with " + str(settings["model_type"]), settings=settings)
        expected_value_viz.setInteractive()
        expected_value_viz.init()
        criticLosses = []
        
    masterAgent.setSettings(settings)
    masterAgent.setExperience(experience)
    masterAgent.setPolicy(model)
    
    # masterAgent.setPolicy(model)
    # masterAgent.setForwardDynamics(forwardDynamicsModel)
    namespace.agentPoly = masterAgent.getPolicy().getNetworkParameters()
    namespace.model = model
    
    # mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModel(actor, exp, masterAgent, discount_factor, anchors=settings['eval_epochs'], 
    #                                                                                                                     action_space_continuous=action_space_continuous, settings=settings, print_data=True, evaluation=True,
    #                                                                                                                   visualizeEvaluation=expected_value_viz)
        # simEpoch(exp, model, discount_factor=discount_factor, anchors=_anchors[:settings['eval_epochs']][9], action_space_continuous=True, settings=settings, print_data=True, p=0.0, validation=True)
    
    for k in range(5):
        mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModelParrallel( input_anchor_queue=input_anchor_queue,
                                                                model=masterAgent, settings=settings, eval_episode_data_queue=eval_episode_data_queue, anchors=settings['eval_epochs'])
    
        print ("Mean eval: ", mean_eval)
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
        exp.init()
        
        w = SimWorker(input_anchor_queue, output_experience_queue, exp, model, discount_factor, action_space_continuous=action_space_continuous, 
                settings=settings, print_data=False, p=0.0, validation=True)
        w.start()
        workers.append(w)
        
    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error = evalModelParrallel(
        input_anchor_queue, output_experience_queue, discount_factor, anchors=_anchors[:settings['eval_epochs']], action_space_continuous=action_space_continuous, settings=settings)
    
    for w in workers:
        input_anchor_queue.put(None)
       """ 
    print ("Mean Evaluation: " + str(mean_eval))
    
    print ("Terminating Workers")
    for sw in sim_workers: # Should update these more offten
        input_anchor_queue.put(None)
        
    for sw in sim_workers: # Should update these more offten
        sw.join()
        
def _modelEvaluation(inputData):
    try:
        modelEvaluation(settings_file_name=inputData[0], settings=inputData[1])
    except:
        print ("model evaluation failed")
        
    return True

def modelEvaluation(settings_file_name, settings=None, runLastModel=False, render=True, exp=None):
    
    from model.ModelUtil import getSettings
    from util.SimulationUtil import setupEnvironmentVariable, setupLearningBackend
    if (settings is None):
        settings = getSettings(settings_file_name)
    # settings['shouldRender'] = True
    setupEnvironmentVariable(settings, eval=True)
    setupLearningBackend(settings)
    ### Flag so simulation models can be a little different.
    settings["simulation_model"] = True
    ## Theano needs to be imported after the flags are set.
    # from ModelEvaluation import *
    # from model.ModelUtil import *
    from simulation.evalModel import evalModelParrallel, evalModel
    # from model.ModelUtil import validBounds
    from model.LearningAgent import LearningAgent, LearningWorker
    from util.SimulationUtil import validateSettings, createEnvironment, createRLAgent, createActor, createNewFDModel
    from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler, getAgentName, processBounds
    
    
    from util.ExperienceMemory import ExperienceMemory
    from RLVisualize import RLVisualize
    from NNVisualize import NNVisualize
    import imageio
    
    # from model.ModelUtil import *
    # from actor.ActorInterface import *
    # from util.SimulationUtil import *
    
    # anchor_data_file = open(settings["anchor_file"])
    # _anchors = getAnchors(anchor_data_file)
    # anchor_data_file.close()
    # settings['shouldRender'] = True 
    model_type= settings["model_type"]
    directory= getDataDirectory(settings)
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    # num_states=settings["num_states"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    # max_reward=settings["max_reward"]
    batch_size=settings["batch_size"]
    state_bounds = settings['state_bounds']
    action_space_continuous=settings["action_space_continuous"]  
    discrete_actions = settings['discrete_actions']
    reward_bounds=np.array(settings["reward_bounds"])
    action_space_continuous=settings['action_space_continuous']
    if ( not (settings["action_bounds"] == "ask_env")) and action_space_continuous:
        action_bounds = settings["action_bounds"]
    else:
        action_bounds = [None]
    
    print ("Sim config file name: " + str(settings["sim_config_file"]))
    sim_index=0
    if ( 'override_sim_env_id' in settings and (settings['override_sim_env_id'] != False)):
        sim_index = settings['override_sim_env_id']
    # exp = createEnvironment(settings["sim_config_file"], settings['environment_type'], settings, render=True, index=sim_index)
    actor = createActor(str(settings['environment_type']),settings, None)
    
    if (exp is None):
        exp = createEnvironment(settings["sim_config_file"], settings['environment_type'], settings, render=render, index=sim_index)
        exp.setActor(actor)
        exp.getActor().init()
        exp.init()
        
    (state_bounds, action_bounds, settings) = processBounds(state_bounds, action_bounds, settings, exp)
    
    if ( "perform_multiagent_training" in settings):
        from model.LearningMultiAgent import LearningMultiAgent
        masterAgent = LearningMultiAgent(settings_=settings)
    else:
        masterAgent = LearningAgent(settings_=settings)
    
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    # print("Loading model: ", file_name)
    # f = open(file_name, 'rb')
    # model = dill.load(f)
    # f.close()
    if (runLastModel is True):
        settings["load_saved_model"] = "last"
    else:
        settings["load_saved_model"] = True
    # settings["load_saved_model"] = "network_and_scales"
    model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
    # print ("State Length: ", len(model.getStateBounds()[0]) )
    
    if ("train_forward_dynamics" in settings and settings['train_forward_dynamics']):
        if (runLastModel == True):
            # createNewFDModel(settings, exp_val, model)
            forwardDynamicsModel = createNewFDModel(settings, exp, model)
            # forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=None, print_info=True)
        else:
            forwardDynamicsModel = createNewFDModel(settings, exp, model)
            # forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=None, print_info=True)
        
        # forwardDynamicsModel.setActor(actor)
        masterAgent.setForwardDynamics(forwardDynamicsModel)

    if ( "use_simulation_sampling" in settings and settings['use_simulation_sampling'] ):
        sampler = createSampler(settings, exp)
        ## This should be some kind of copy of the simulator not a network
        if (not settings['train_forward_dynamics']):
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, actor, exp, agentModel=None, print_info=True)
        sampler.setForwardDynamics(forwardDynamicsModel)
        # sampler.setPolicy(model)
        masterAgent.setSampler(sampler)
        # print ("thread together exp: ", masterAgent._exp)
        # sys.exit()
            
    if ( "use_transfer_task_network" in settings and settings["use_transfer_task_network"] ):
        task_directory = getTaskDataDirectory(settings)
        file_name=directory+getAgentName()+"_Best.pkl"
        f = open(file_name, 'rb')
        taskModel = dill.load(f)
        f.close()
        # copy the task part from taskModel to model
        print ("Transferring task portion of model.")
        model.setTaskNetworkParameters(taskModel)

    exp.setActor(actor)
    exp.getActor().init()   
    exp.init()
    expected_value_viz=None
    if ("visualize_expected_value" in settings and (settings['visualize_expected_value'] == True)):
        expected_value_viz = NNVisualize(title=str("Reward"), settings=settings, nice=True)
        expected_value_viz.setInteractive()
        expected_value_viz.init()
        criticLosses = []

    """
    if ("perform_multiagent_training" in settings):
        experience = [ExperienceMemory(len(state_bounds[i][0]), len(action_bounds[i][0]),
                                       settings['experience_length'][i], continuous_actions=True, settings=settings)
                      for i in range(settings["perform_multiagent_training"])]
    else:
        ### Using a wrapper for the type of actor now
        if action_space_continuous:
            experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['experience_length'],
                                          continuous_actions=True, settings=settings)
        else:
            experience = ExperienceMemory(len(state_bounds[0]), 1, settings['experience_length'])
        # actor = ActorInterface(discrete_actions)
    """
        
    masterAgent.setSettings(settings)
    # masterAgent.setExperience(experience)
    masterAgent.setPolicy(model)
    
    # print (masterAgent.getRewardModel())
    # sys.exit()
    
    movieWriter = None
    if ("save_video_to_file" in settings):
        movieWriter = imageio.get_writer(directory + settings["save_video_to_file"], mode='I',  fps=30)
    
    
    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval, otherMetrics = evalModel(actor, exp, masterAgent, discount_factor, anchors=settings['eval_epochs'], 
                                                                                                                        action_space_continuous=action_space_continuous, settings=settings, print_data=True, evaluation=True,
                                                                                                                        visualizeEvaluation=expected_value_viz, movieWriter=movieWriter)
        # simEpoch(exp, model, discount_factor=discount_factor, anchors=_anchors[:settings['eval_epochs']][9], action_space_continuous=True, settings=settings, print_data=True, p=0.0, validation=True)
    
    print ("otherMetrics: ", otherMetrics)
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
        exp.init()
        
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
    evalData = {}
    evalData['mean_reward'] = mean_reward
    evalData['std_reward'] = std_reward
    evalData['mean_bellman_error'] = mean_bellman_error
    evalData['std_bellman_error'] = std_bellman_error
    evalData['mean_discount_error'] = mean_discount_error
    evalData['std_discount_error'] = std_discount_error
    evalData['mean_eval'] = mean_eval
    evalData['std_eval'] = std_eval
    evalData.update(otherMetrics)
    evalData.update(settings)
    fp = open(directory+"evalData_" + str(settings['agent_name']) + ".json", 'w')
    from util.utils import NumpyEncoder 
    # print ("trainData: ", trainData)
    json.dump(evalData, fp, cls=NumpyEncoder)
    fp.close()
    evalData['masterAgent'] = masterAgent
    exp.finish()
    if ("save_video_to_file" in settings):
        movieWriter.close()
    if (settings['visualize_expected_value'] == True):
        expected_value_viz.finish()
    
    return evalData
    
if __name__ == "__main__":
    """
        If a third param is specified run in the last saved model not the best model.
    """
    import time
    import datetime
    from util.simOptions import getOptions
    
    options = getOptions(sys.argv)
    options = vars(options)
    print("options: ", options)
    print("options['configFile']: ", options['configFile'])
        
    
    
    file = open(options['configFile'])
    settings = json.load(file)
    file.close()
    
    for option in options:
        if ( not (options[option] is None) ):
            print ("Updating option: ", option, " = ", options[option])
            settings[option] = options[option]
            if ( options[option] == 'true'):
                settings[option] = True
            elif ( options[option] == 'false'):
                settings[option] = False
        # settings['num_available_threads'] = options['num_available_threads']

    print ("Settings: " + str(json.dumps(settings, indent=4)))
    
    if (settings['shouldRender'] == 'yes'):
        sim = modelEvaluation(sys.argv[1], runLastModel=True, settings=settings, render='yes')
    else:
        sim = modelEvaluation(sys.argv[1], runLastModel=True, settings=settings, render=settings["shouldRender"])
    
