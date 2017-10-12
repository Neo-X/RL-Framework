

from trainModel import trainModelParallel
import sys
import json
import copy
from pathos.threading import ThreadPool
from pathos.multiprocessing import ProcessingPool
# from threading import ThreadPool
import time
import datetime

from util.SimulationUtil import getDataDirectory, getBaseDataDirectory

def _trainMetaModel(input):
    settingsFileName_ = input[0]
    samples_ = input[1]
    settings_ = input[2]
    numThreads_ = input[3]
    if (len(input) > 4 ):
        hyperSettings_ = input[4]
        return trainMetaModel(settingsFileName_, samples=samples_, settings=settings_, numThreads=numThreads_, 
                              hyperSettings=hyperSettings_)
    else:
        return trainMetaModel(settingsFileName_, samples=samples_, settings=settings_, numThreads=numThreads_)
    
    
def trainMetaModel(settingsFileName, samples=10, settings=None, numThreads=1, HyperSettings=None):
    import shutil
    import os
    
    if (settings is None):
        file = open(settingsFileName)
        settings = json.load(file)
        print ("Settings: " + str(json.dumps(settings)))
        file.close()
    
    print ( "Running ", samples, " simulation(s) over ", numThreads, " Thread(s)")
    data_name = settings['data_folder']
    sim_settings=[]
    sim_settingFileNames=[]
    sim_data = []
    for i in range(samples):
        settings['data_folder'] = data_name + "_" + str(i)
        settings['random_seed'] = int(settings['random_seed']) + ((int(settings['num_available_threads']) + 1) * i)
        ## Change some other settings to reduce memory usage and train faster
        settings['print_level'] = "hyper_train"
        settings['shouldRender'] = False
        settings['visualize_learning'] = False
        
        sim_settings.append(copy.deepcopy(settings))
        sim_settingFileNames.append(settingsFileName)
        sim_data.append((settingsFileName,copy.deepcopy(settings)))
        
        ## Create data directory and copy any desired files to these folders .
        if ( not (HyperSettings is None) ):
            directory= getDataDirectory(settings)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if ('saved_fd_model_path' in HyperSettings):
                print ("Copying fd model: ", HyperSettings['saved_fd_model_path'])
                shutil.copy2(HyperSettings['saved_fd_model_path'], directory+"forward_dynamics_"+str(settings['agent_name'])+"_Best_pretrain.pkl" )
        
    # p = ThreadPool(numThreads)
    p = ProcessingPool(numThreads)
    t0 = time.time()
    result = p.map(trainModelParallel, sim_data)
    t1 = time.time()
    print ("Meta model training complete in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds")
    print (result)
    # trainModelParallel(settingsFileName, copy.deepcopy(settings))
        
    

if (__name__ == "__main__"):
    """
        python trainMetaModel.py <hyper_settings_file> <sim_settings_file> <num_samples> <num_threads> <saved_fd_model_path>
        Example:
        python trainMetaModel.py settings/navGame/PPO_5D.json 10
    """
    from sendEmail import sendEmail
    import json
    
    if (len(sys.argv) == 1):
        print("Please incluse sim settings file")
        print("python trainMetaModel.py <hyper_settings_file> <sim_settings_file> <num_samples>")
        sys.exit()
    elif (len(sys.argv) == 2):
        print("Please incluse sim settings file")
        print("python trainMetaModel.py <hyper_settings_file> <sim_settings_file> <num_samples>")
        sys.exit()
    elif (len(sys.argv) == 3):
        print("Please incluse sim settings file")
        print("python trainMetaModel.py <hyper_settings_file> <sim_settings_file> <num_samples>")
        sys.exit()
    elif (len(sys.argv) == 4):
        
        settingsFileName = sys.argv[1] 
        file = open(settingsFileName)
        settings_ = json.load(file)
        print ("Settings: " + str(json.dumps(settings_)))
        file.close()
        
        trainMetaModel(sys.argv[2], samples=int(sys.argv[3]))
        ## Send an email so I know this has completed
        sendEmail("Simulation complete", sys.argv[1], settings=settings)
    elif (len(sys.argv) == 5):
        settingsFileName = sys.argv[1] 
        file = open(settingsFileName)
        settings_ = json.load(file)
        print ("Settings: " + str(json.dumps(settings_)))
        file.close()
        
        trainMetaModel(sys.argv[2], samples=int(sys.argv[3]), numThreads=int(sys.argv[4]))
        ## Send an email so I know this has completed
        sendEmail("Simulation complete", sys.argv[1], settings=settings_)    
    elif (len(sys.argv) == 6):
        settingsFileName = sys.argv[1] 
        file = open(settingsFileName)
        settings_ = json.load(file)
        print ("Settings: " + str(json.dumps(settings_)))
        file.close()
        settings_['saved_fd_model_path'] = sys.argv[5]
        trainMetaModel(sys.argv[2], samples=int(sys.argv[3]), numThreads=int(sys.argv[4]), HyperSettings=settings_)
        ## Send an email so I know this has completed
        sendEmail("Simulation complete", sys.argv[1], settings=settings_)      
    else:
        print("Please specify arguments properly, ")
        print(sys.argv)
        print("python trainMetaModel.py <hyper_settings_file> <sim_settings_file> <num_samples>")
        
