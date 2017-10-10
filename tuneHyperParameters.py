

from trainMetaModel import trainMetaModel, _trainMetaModel
import sys
import json
import copy
from pathos.threading import ThreadPool
from pathos.multiprocessing import ProcessingPool
import time
import datetime

from util.SimulationUtil import getDataDirectory, getBaseDataDirectory
"""
def tuneHyperParameters(simsettingsFileName, Hypersettings=None):
"""
        # For some set of parameters the functino will sample a number of them
        # In order to find a more optimal configuration.
"""
    import os
    num_sim_samples=5
    file = open(simsettingsFileName)
    settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings, indent=4)))
    file.close()
    samples = 5
    param_of_interest = 'action_learning_rate'
    range_ = [0.05, 1.0]
    data_name = settings['data_folder']
    for i in range(samples+1):
        param_value = ((range_[1] - range_[0]) * (float(i)/samples)) + range_[0]
        settings['data_folder'] = data_name + "_" + str(param_value) + "/"
        settings['action_learning_rate'] = param_value
        directory= getBaseDataDirectory(settings)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # file = open(settingsFileName, 'r')
        out_file_name=directory+os.path.basename(simsettingsFileName)
        
        print ("Saving settings file with data to: ", out_file_name)
        out_file = open(out_file_name, 'w')
        out_file.write(json.dumps(settings, indent=4))
        # file.close()
        out_file.close()
        
        trainMetaModel(simsettingsFileName, samples=num_sim_samples, settings=copy.deepcopy(settings), numThreads=num_sim_samples)
"""

def tuneHyperParameters(simsettingsFileName, Hypersettings=None):
    """
        For some set of parameters the function will sample a number of them
        In order to find a more optimal configuration.
    """
    import os
    file = open(simsettingsFileName)
    settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings, indent=4)))
    file.close()
    file = open(Hypersettings)
    hyper_settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings, indent=4)))
    file.close()
    num_sim_samples = hyper_settings['meta_sim_samples']
    
    ## Check to see if there exists a saved fd model, if so save the path in the hyper settings
    directory= getDataDirectory(settings)
    file_name_dynamics=directory+"forward_dynamics_"+str(settings['agent_name'])+"_Best_pretrain.pkl" 
    if not os.path.exists(directory):
        Hypersettings['saved_fd_model_path'] = file_name_dynamics
            
    
    samples = hyper_settings['num_param_samples'] - 1
    param_of_interest = hyper_settings['param_to_tune']
    range_ = hyper_settings['param_bounds']
    data_name = settings['data_folder']
    sim_data = []
    for i in range(samples+1):
        param_value = ((range_[1] - range_[0]) * (float(i)/samples)) + range_[0]
        settings['data_folder'] = data_name + "_" + param_of_interest + "_"+ str(param_value) + "/"
        settings[param_of_interest] = param_value
        directory= getBaseDataDirectory(settings)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # file = open(settingsFileName, 'r')
        out_file_name=directory+os.path.basename(simsettingsFileName)
        
        print ("Saving settings file with data to: ", out_file_name)
        out_file = open(out_file_name, 'w')
        out_file.write(json.dumps(settings, indent=4))
        # file.close()
        out_file.close()
        sim_data.append((simsettingsFileName, num_sim_samples, copy.deepcopy(settings), hyper_settings['meta_sim_threads'], copy.deepcopy(Hypersettings)))
        
    
    # p = ProcessingPool(2)
    p = ThreadPool(hyper_settings['tuning_threads'])
    t0 = time.time()
    result = p.map(_trainMetaModel, sim_data)
    t1 = time.time()
    print ("Hyper parameter tuning complete in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds")
    print (result)
    

if (__name__ == "__main__"):
    """
        python tuneHyperParameters.py <sim_settings_file> <tuning_settings_file>
        Example:
        python tuneHyperParameters.py settings/navGame/PPO_5D.json settings/navGame/PPO_5D_hyper.json 
    """
    
    if (len(sys.argv) == 1):
        print("Please incluse sim settings file")
        print("python tuneHyperParameters.py <sim_settings_file> <tuning_settings_file>")
        sys.exit()
    elif (len(sys.argv) == 2):
        print("Please incluse sim settings file")
        print("python tuneHyperParameters.py <sim_settings_file> <tuning_settings_file>")
        sys.exit()
    elif (len(sys.argv) == 3):
        tuneHyperParameters(sys.argv[1], sys.argv[2])
    else:
        print("Please specify arguments properly, ")
        print(sys.argv)
        print("python tuneHyperParameters.py <sim_settings_file> <tuning_settings_file>")
        
        
        