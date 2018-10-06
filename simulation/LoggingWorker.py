
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

def collectEmailData(settings, metaSettings, sim_time_=0, simData={}):
    from sendEmail import sendEmail
    import json
    import tarfile
    from util.SimulationUtil import addDataToTarBall, addPicturesToTarBall
    from util.SimulationUtil import getDataDirectory, getBaseDataDirectory, getRootDataDirectory, getAgentName
    import os
    
    ### Create a tar file of all the sim data
    root_data_dir = getDataDirectory(settings)+"/"
    tarFileName = (root_data_dir + '_sim_data.tar.gz_') ## gmail doesn't like compressed files....so change the file name ending..
    dataTar = tarfile.open(tarFileName, mode='w:gz')
    addDataToTarBall(dataTar, settings)
        
    print("root_data_dir: ", root_data_dir)
    pictureFileName=None
    try:
        ## Add pictures to tar file
        _data_dir = getDataDirectory(settings)
        addPicturesToTarBall(dataTar, settings, data_folder=_data_dir)
        pictureFileName=  root_data_dir + getAgentName() + ".png"
    except Exception as e:
        # dataTar.close()
        print("Error plotting data there my not be a DISPLAY available.")
        print("Error: ", e)
    dataTar.close()
    
    
    ## Send an email so I know this training has completed
    contents_ = json.dumps(metaSettings, indent=4, sort_keys=True)
    sub = "Simulation complete: " + str(sim_time_)
    simData = {}
    if ('error' in simData):
        contents_ = contents_ + "\n" + simData['error']
        sub = "ERROR*****     " + "Simulation terminated: " + str(sim_time_)
     
    sendEmail(subject=sub, contents=contents_, hyperSettings=metaSettings, simSettings=settings['configFile'], dataFile=tarFileName,
              pictureFile=pictureFileName) 


# class SimWorker(threading.Thread):
class LoggingWorker(Process):
    
    def __init__(self,  
                 settings, loggingWorkerQueue):
        super(LoggingWorker, self).__init__()
        self._settings= settings
        metaSettings = None
        if ( 'metaConfigFile' in settings and (settings['metaConfigFile'] is not None)):
            ### Import meta settings
            file = open(settings['metaConfigFile'])
            metaSettings = json.load(file)
            file.close()
            self._metaSettings = metaSettings
        self._loggingWorkerQueue = loggingWorkerQueue
        
    # @profile(precision=5)
    def run(self):
        
        # timeout_ = 60 * 60 * 12 ### time between data emails.
        timeout_ = 60  ### time between data emails.
        steps__ = 0
        timesteps = 0
        running = True
        while (running):
            time.sleep(1)
            # try:
            if ( steps__ > timeout_ ):
                print("Sending log email after ", timeout_, " seconds")
                collectEmailData(self._settings, self._metaSettings, sim_time_=timesteps)
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
        