import sys
import numpy as np
import dill
import dill as pickle
import dill as cPickle

def combineNetworkModels(settings_file_name):
    
    from model.ModelUtil import getSettings
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
    from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler
    
    
    from util.ExperienceMemory import ExperienceMemory
    from RLVisualize import RLVisualize
    from NNVisualize import NNVisualize
    
    directory= getDataDirectory(settings)
    
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    # num_states=settings["num_states"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    reward_bounds=np.array(settings["reward_bounds"])
    # reward_bounds = np.array([[-10.1],[0.0]])
    batch_size=settings["batch_size"]
    train_on_validation_set=settings["train_on_validation_set"]
    state_bounds = np.array(settings['state_bounds'])
    discrete_actions = np.array(settings['discrete_actions'])
    num_actions= discrete_actions.shape[0] # number of rows
    print ("Sim config file name: " + str(settings["sim_config_file"]))
    # c = characterSim.Configuration(str(settings["sim_config_file"]))
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    action_space_continuous=settings['action_space_continuous']
    
    new_model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
    
    if (False):
        file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+".pkl"
    else:
        file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"_Best.pkl"
    print("Loading model: ", file_name)
    f = open(file_name, 'rb')
    old_model = dill.load(f)
    f.close()
    print ("State Length: ", len(old_model.getStateBounds()[0]) )
    
    new_model.setAgentNetworkParamters(old_model)
    new_model.setCombinedNetworkParamters(old_model)
    # new_model.setNetworkParameters(old_model.getNetworkParameters())
    
    file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"_Injected.pkl"
    f = open(file_name, 'wb')
    dill.dump(new_model, f)
    f.close()
    
    """
    ### Want to copy parts of  old model over new model
    if ( settings["use_transfer_task_network"] ):
        task_directory = getTaskDataDirectory(settings)
        file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"_Best.pkl"
        f = open(file_name, 'rb')
        taskModel = dill.load(f)
        f.close()
        # copy the task part from taskModel to model
        print ("Transferring task portion of model.")
        model.setTaskNetworkParameters(taskModel)

    """

if __name__ == '__main__':


    combineNetworkModels(sys.argv[1])