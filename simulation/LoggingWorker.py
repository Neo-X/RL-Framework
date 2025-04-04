
# import cPickle
import dill
import sys
import gc
# from theano.compile.io import Out
sys.setrecursionlimit(50000)
# from sim.PendulumEnvState import PendulumEnvState
# from sim.PendulumEnv import PendulumEnv
from multiprocessing import Process, Queue
import logging
import multiprocessing
# from pathos.multiprocessing import Pool
import time
import copy
import json
import os
import queue
# import memory_profiler
# import resources

log = logging.getLogger(os.path.basename(__file__))

# class SimWorker(threading.Thread):
class LoggingWorker(Process):
    
    def __init__(self,  
                 settings,
                 emailFunction, 
                 loggingWorkerQueue,
                 simData={}):
        super(LoggingWorker, self).__init__()
        self._settings= settings
        self._emailFunction = emailFunction
        self._simData = simData
        self._metaSettings = None
        if ( 'metaConfigFile' in settings and (settings['metaConfigFile'] is not None)):
            ### Import meta settings
            file = open(settings['metaConfigFile'])
            metaSettings = json.load(file)
            file.close()
            self._metaSettings = metaSettings
        self._loggingWorkerQueue = loggingWorkerQueue
        
    # @profile(precision=5)
    def run(self):
        from util.SimulationUtil import setupEnvironmentVariable, setupLearningBackend
        timeout_ = 60 * 60 * 8 ### time between data emails (8 hours).
        # timeout_ = 60  ### time between data emails.
        if ( "email_logging_time" in self._settings ):
            timeout_ = self._settings["email_logging_time"]
        steps__ = 0
        timesteps = 0
        exp = None
        setupEnvironmentVariable(self._settings, eval=True)
        import numpy as np
        if ("save_video_to_file" in self._settings):
            from util.SimulationUtil import createEnvironment
            ### need to create and keep around and reuse a pointer to a simulation because glut is a pain in the butt...
            exp = createEnvironment(self._settings["sim_config_file"], self._settings['environment_type'], self._settings, render=True, index=0)
            vizData = exp.getEnvironment().render()
            # movie_writer.append_data(np.transpose(vizData))
#             print ("**********************************************sim image mean: ", np.mean(vizData), " std: ", np.std(vizData))
            if ("test_movie_rendering" in self._settings
                and (self._settings["test_movie_rendering"] == True)):
                return
            # pass
        running = True
        data = None
        while (running):
            try:
                ### Check if done first
                data_ = self._loggingWorkerQueue.get(timeout=1) ### 1 second timeout
                if type(data_) is tuple:
                    # Data format: (STRING:type of information, <any type>: data)
                    if data_[0] == "checkpoint_vid_rounds":
                        from ModelEvaluation import modelEvaluation
                        roundNum = data_[1]
#                         log.info('Creating video for checkpoint round {}'.format(roundNum))
                        settings_copy = copy.deepcopy(self._settings)
                        filename = settings_copy['save_video_to_file']
                        settings_copy['save_video_to_file'] = filename[:filename.rindex('.')] + '_round' + str(roundNum) + filename[filename.rindex('.'):]
                        modelEvaluation("", settings=settings_copy, exp=exp) # Save a video for this checkpoint
                else:
                    running = running and data_
                    if (not running):
                        break
            except (queue.Empty, OSError) as error:
#                 log.warning("Caught error when attempting to evaluate model: {}".format(error))
                pass
            time.sleep(1)
            # try:
            if ( ( steps__ >= timeout_ )
                 and 
                   (self._metaSettings is not None)
                 ):
                print("Sending log email after ", timeout_, " seconds")
                
                if ("save_video_to_file" in self._settings
                    and (exp is not None)):
                    print("Saving video email function calling: ", exp)
                    self._emailFunction(self._settings, self._metaSettings, sim_time_=timesteps, simData=self._simData, exp=exp)
                else:
                    self._emailFunction(self._settings, self._metaSettings, sim_time_=timesteps, simData=self._simData)
                    
                steps__ = 0
                
            steps__ = steps__ + 2
            timesteps = timesteps + 2
            # except:
            #     print ("Failed emailing log data.")

        if ("save_video_to_file" in self._settings):
            print("Saving video email function calling: ", exp)
            self._emailFunction(self._settings, self._metaSettings, sim_time_=timesteps, simData=self._simData, exp=exp)
        print ("Terminating logging emailing process.")
        
