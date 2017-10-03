

from trainMetaModel import trainMetaModel, _trainMetaModel
import sys
import json
import copy
from pathos.threading import ThreadPool
from pathos.multiprocessing import ProcessingPool

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
        For some set of parameters the functino will sample a number of them
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
    
    samples = hyper_settings['num_param_samples'] - 1
    param_of_interest = hyper_settings['param_to_tune']
    range_ = hyper_settings['param_bounds']
    data_name = settings['data_folder']
    sim_data = []
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
        sim_data.append((simsettingsFileName, num_sim_samples, copy.deepcopy(settings), hyper_settings['meta_sim_threads']))
        
    
    # p = ProcessingPool(2)
    p = ThreadPool(2)
    result = p.map(_trainMetaModel, sim_data)
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
        
        
        