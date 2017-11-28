import sys

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
    
    new_model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
    
    if (False):
        file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+".pkl"
    else:
        file_name=directory+"pendulum_agent_"+str(settings['agent_name'])+"_Best.pkl"
    print("Loading model: ", file_name)
    f = open(file_name, 'rb')
    old_model = dill.load(f)
    f.close()
    print ("State Length: ", len(model.getStateBounds()[0]) )
    
    new_model.setAgentNetworkParamters(old_model)
    
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


    