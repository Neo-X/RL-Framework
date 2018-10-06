
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
import time
import copy
import json
# import memory_profiler
# import resources


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
        
        # timeout_ = 60 * 60 * 12 ### time between data emails (12 hours).
        timeout_ = 60  ### time between data emails.
        if ( "email_logging_time" in self._settings ):
            timeout_ = self._settings["email_logging_time"]
        steps__ = 0
        timesteps = 0
        running = True
        while (running):
            time.sleep(1)
            # try:
            if ( steps__ >= timeout_ and (self._metaSettings is not None)):
                print("Sending log email after ", timeout_, " seconds")
                self._emailFunction(self._settings, self._metaSettings, sim_time_=timesteps, simData=self._simData)
                steps__ = 0
                
            steps__ = steps__ + 1
            timesteps = timesteps + 1
            try:
                # print ("Getting email queue data")
                data_ = self._loggingWorkerQueue.get(False)
                # print ("email worker data: ", data_)
                running = running and data_
            except Exception as inst:
                # print ("SimWorker model parameter message queue empty.")
                pass
            # except:
            #     print ("Failed emailing log data.")

        print ("Terminating logging emailing process.")
        