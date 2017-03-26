

from model.ModelUtil import *
# import cPickle
import dill
import sys
# from theano.compile.io import Out
sys.setrecursionlimit(50000)
from sim.PendulumEnvState import PendulumEnvState
from sim.PendulumEnv import PendulumEnv
from multiprocessing import Process, Queue
# from pathos.multiprocessing import Pool
import threading
import time
import copy

from actor.ActorInterface import *

import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

exp=None

fps=30
class SimContainer(object):
    
    def __init__(self, exp, agent):
        self._exp = exp
        self._agent = agent
        self._episode=0
        
    def animate(self, callBackVal=-1):
        # print ("Animating: ", callBackVal)
        """
        glClearColor(0.8, 0.8, 0.9, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
    
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective (45.0, 1.3333, 0.2, 20.0)
    
        glViewport(0, 0, 640, 480)
    
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    
        glLightfv(GL_LIGHT0,GL_POSITION,[0, 0, 1, 0])
        glLightfv(GL_LIGHT0,GL_DIFFUSE,[1, 1, 1, 1])
        glLightfv(GL_LIGHT0,GL_SPECULAR,[1, 1, 1, 1])
        glEnable(GL_LIGHT0)
    
        glEnable(GL_COLOR_MATERIAL)
        glColor3f(0.8, 0.8, 0.8)
    
        gluLookAt(1.5, 4.0, 3.0, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0)
        
        glutPostRedisplay()
        
        """
        
        # print ("End of Epoch: ", self._exp.getEnvironment().endOfEpoch())
        if (self._exp.getEnvironment().endOfEpoch() and 
               self._exp.getEnvironment().needUpdatedAction()):
            self._exp.getActor().initEpoch()
            self._exp.generateValidation(10, self._episode)
            self._exp.getEnvironment().initEpoch()
            self._episode += 1
            
        glutTimerFunc(1000/fps, self.animate, 0) # 30 fps?
        
        if (self._exp.getEnvironment().needUpdatedAction()):
            state_ = self._exp.getState()
            action_ = np.array(self._agent.predict(state_), dtype='float64')
            print( "New action: ", action_)
            self._exp.getEnvironment().updateAction(action_)
        
        self._exp.update()
        
    def onKey(self, c, x, y):
        """GLUT keyboard callback."""
    
        global SloMo, Paused
    
        # set simulation speed
        if c >= '0' and c <= '9':
            SloMo = 4 * int(c) + 1
        # pause/unpause simulation
        elif c == 'p' or c == 'P':
            Paused = not Paused
        # quit
        elif c == 'q' or c == 'Q':
            sys.exit(0)
        elif c == 'r':
            print("Resetting Epoch")
            self._exp.getActor().initEpoch()   
            self._exp.getEnvironment().initEpoch()

def evaluateModelRender(settings_file_name):

    settings = getSettings(settings_file_name)
    # settings['shouldRender'] = True
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
    
    from util.SimulationUtil import validateSettings, createEnvironment, createRLAgent, createActor
    from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel
    from util.ExperienceMemory import ExperienceMemory
    from model.LearningAgent import LearningAgent, LearningWorker
    from RLVisualize import RLVisualize
    from NNVisualize import NNVisualize
    
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
    
    exp = createEnvironment(str(settings["sim_config_file"]), str(settings['environment_type']), settings, render=True)
    if (settings['train_forward_dynamics']):
        # actor.setForwardDynamicsModel(forwardDynamicsModel)
        forwardDynamicsModel.setActor(actor)
        masterAgent.setForwardDynamics(forwardDynamicsModel)
        # forwardDynamicsModel.setEnvironment(exp)
    # actor.setPolicy(model)
    
    exp.getActor().init()   
    exp.getEnvironment().init()
    exp.generateValidationEnvironmentSample(0)
    expected_value_viz=None
    if (settings['visualize_expected_value']):
        expected_value_viz = NNVisualize(title=str("Expected Value") + " with " + str(settings["model_type"]), settings=settings)
        expected_value_viz.setInteractive()
        expected_value_viz.init()
        criticLosses = []
        
    masterAgent.setSettings(settings)
    masterAgent.setExperience(experience)
    masterAgent.setPolicy(model)
    
    """
    mean_reward, std_reward, mean_bellman_error, std_bellman_error, mean_discount_error, std_discount_error, mean_eval, std_eval = evalModel(actor, exp, masterAgent, discount_factor, anchors=_anchors[:settings['eval_epochs']], 
                                                                                                                        action_space_continuous=action_space_continuous, settings=settings, print_data=True, evaluation=True,
                                                                                                                        visualizeEvaluation=expected_value_viz)
        # simEpoch(exp, model, discount_factor=discount_factor, anchors=_anchors[:settings['eval_epochs']][9], action_space_continuous=True, settings=settings, print_data=True, p=0.0, validation=True)
    """
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
    # print ("Average Reward: " + str(mean_reward))
    
    exp.getActor().initEpoch()   
    exp.getEnvironment().initEpoch()
    fps=30
    state_ = exp.getState()
    action_ = np.array(masterAgent.predict(state_), dtype='float64')
    exp.getEnvironment().updateAction(action_)
    sim = SimContainer(exp, masterAgent)
    # glutInitWindowPosition(x, y);
    # glutInitWindowSize(width, height);
    # glutCreateWindow("PyODE Ragdoll Simulation")
    # set GLUT callbacks
    glutKeyboardFunc(sim.onKey)
    ## This works because GLUT in C++ uses the same global context (singleton) as the one in python 
    glutTimerFunc(1000/fps, sim.animate, 0) # 30 fps?
    # glutIdleFunc(animate)
    # enter the GLUT event loop
    glutMainLoop()
    
    
if __name__ == "__main__":
    
    evaluateModelRender(sys.argv[1])
