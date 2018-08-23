

from trainMetaModel import trainMetaModel, _trainMetaModel
import sys
import json
import copy
from pathos.threading import ThreadPool
import time
import datetime
from tools.PlotMetadataSimulation import plotMetaDataSimulation

from util.SimulationUtil import getDataDirectory, getBaseDataDirectory, getRootDataDirectory
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

def compute_next_val(range_,i,samples, curve_scheme='linear'):
    """
    
    """
    if ( (curve_scheme == 'linear')):
        delta_ = ((float(i)) / float(samples))
    elif (curve_scheme == "squared"):
        delta_ = ((float(i)) / float(samples))
        delta_ = delta_**2
    elif (curve_scheme == "exponential"):
        delta_ = ((float(i)) / float(samples))
        delta_ = delta_**(samples-i) 
        
    delta_ = delta_ * (range_[1] - range_[0]) 
    return delta_

def makeNiceName(params_to_tune):
    """
        Take the list of parameters to sample over and return a nice
        string that will result in a good filename.
    """
    out = ""
    for p in params_to_tune:
        out = out + "_" + str(p)
    return out

def get_param_values(hyper_settings):
    """
        Returns the cross product of the parameters
    """
    import itertools
    
    parameter_samples = hyper_settings['num_param_samples']
    params_ = []
    for par in range(len(parameter_samples)):
        param_of_interest = hyper_settings['param_to_tune'][par]
        range_ = hyper_settings['param_bounds'][par]
        ### Assumes value is numeric..
        if type(hyper_settings['num_param_samples'][par]) is list:
            samples = len(hyper_settings['num_param_samples'][par]) - 1
        else:
            samples = hyper_settings['num_param_samples'][par] - 1
        # data_name = settings['data_folder']
        # sim_data = []
        # result_data['hyper_param_settings_files'] = []
        params_tmp = []
        for i in range(samples+1):
            if (hyper_settings['param_data_type'][par] == "int"):
                param_value = int( ((range_[1] - range_[0]) * (float(i)/samples)) + range_[0] )
            elif (hyper_settings['param_data_type'][par] == "bool"):
                if ( i == 0):
                    param_value = True
                elif ( i == 1):
                    param_value = False
                else:
                    print("Error to many samples for bool type:")
                    sys.exit()
            elif (hyper_settings['param_data_type'][par] == "set"):
                param_value = hyper_settings['num_param_samples'][par][i]
            else: #float
                delta_ = compute_next_val(range_, i, samples, curve_scheme=hyper_settings['curve_scheme'][par])
                # print ("detla: ", delta_)
                param_value = (delta_) + range_[0]
            params_tmp.append(param_value)
        params_.append(params_tmp)
        
    
    print ("params_: ", params_)
    
    if ( len(params_) > 1 ):
        params_ = list(itertools.product(*params_))
    else:
        params__ = []
        for pars in params_[0]:
            params__.append([pars])
        params_ = params__
    # print ("cross other: ", list(itertools.product(*params_)) )
        
    print ("Cross product of params: ", params_)
    # sys.exit()
    return params_
    

def tuneHyperParameters(simsettingsFileName, simSettings, hyperSettings=None, saved_fd_model_path=None):
    """
        For some set of parameters the function will sample a number of them
        In order to find a more optimal configuration.
    """
    import os
    
    result_data = {}
    
    settings = simSettings
    hyper_settings = hyperSettings
    num_sim_samples = hyper_settings['meta_sim_samples']
    
    ## Check to see if there exists a saved fd model, if so save the path in the hyper settings
    if ( not ( saved_fd_model_path is None )):
        directory= getDataDirectory(settings)
        # file_name_dynamics=directory+"forward_dynamics_"+"_Best_pretrain.pkl" 
        if not os.path.exists(directory):
            hyper_settings['saved_fd_model_path'] = saved_fd_model_path
            
    
    param_settings = get_param_values(hyper_settings)
    result_data['hyper_param_settings_files'] = []
    sim_data = []
    data_name = settings['data_folder']
    meta_thread_index = 0
    for params in param_settings: ## Loop over each setting of parameters
        data_name_tmp = ""
        for par in range(len(params)): ## Assemble the vector of parameters and data folder name
            param_of_interest = hyper_settings['param_to_tune'][par]
            data_name_tmp = data_name_tmp + "/_" + param_of_interest + "_"+ str(params[par]) + "/"
            settings[param_of_interest] = params[par]
        
        settings['data_folder'] = data_name + data_name_tmp
        settings['meta_thread_index'] = meta_thread_index
        meta_thread_index = meta_thread_index + 1
        if (meta_thread_index > int(hyper_settings['tuning_threads'])):
            meta_thread_index = 0 
        directory= getBaseDataDirectory(settings)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # file = open(settingsFileName, 'r')
        
        out_file_name=directory+os.path.basename(simsettingsFileName)
        result_data['hyper_param_settings_files'].append(out_file_name)
        print ("Saving settings file with data to: ", out_file_name)
        print ("settings['data_folder']: ", settings['data_folder'])
        out_file = open(out_file_name, 'w')
        out_file.write(json.dumps(settings, indent=4))
        # file.close()
        
        out_file.close()
        sim_data.append((simsettingsFileName, num_sim_samples, copy.deepcopy(settings), hyper_settings['meta_sim_threads'], copy.deepcopy(hyper_settings)))
    
    # p = ProcessingPool(2)
    p = ThreadPool(hyper_settings['tuning_threads'], maxtasksperchild=1)
    t0 = time.time()
    result = p.map(_trainMetaModel, sim_data)
    t1 = time.time()
    print ("Hyper parameter tuning complete in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds")
    result_data['sim_time'] = "Meta model training complete in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds"
    result_data['meta_sim_result'] = result
    result_data['raw_sim_time_in_seconds'] = t1-t0
    result_data['Number_of_simulations_sampled'] = len(param_settings)
    result_data['Number_of_threads_used'] = hyper_settings['tuning_threads'] 
    print (result)
    return result_data
    

if (__name__ == "__main__"):
    """
        python tuneHyperParameters.py <sim_settings_file> <tuning_settings_file>
        Example:
        python tuneHyperParameters.py settings/navGame/PPO_5D.json settings/navGame/PPO_5D_hyper.json 
    """
    import tarfile
    from util.SimulationUtil import addDataToTarBall, addPicturesToTarBall
    from sendEmail import sendEmail
    from util.simOptions import getOptions
    
    options = getOptions(sys.argv)
    options = vars(options)
    
    if ((options['configFile'] == None) 
        or (options['metaConfigFile'] == None)):
        print("Please include sim settings file: ", len(sys.args))
        print("python tuneHyperParameters.py --config=<sim_settings_file> --metaConfig=<hyper_settings_file> --meta_sim_samples=<num_samples> --meta_sim_threads<num_threads>")
        sys.exit()
    else:

        file = open(options['configFile'])
        simSettings_ = json.load(file)
        file.close()
        
        for option in options:
            if ( not (options[option] is None) ):
                print ("Updating option: ", option, " = ", options[option])
                simSettings_[option] = options[option]
                if ( options[option] == 'true'):
                    simSettings_[option] = True
                elif ( options[option] == 'false'):
                    simSettings_[option] = False
            # settings['num_available_threads'] = options['num_available_threads']
        
        # print ("Settings: " + str(json.dumps(settings, indent=4)))
        hyperSettings_ = None
        ### Import meta settings
        file = open(simSettings_['metaConfigFile'])
        hyperSettings_ = json.load(file)
        file.close()
        for option in ['meta_sim_threads', 'tuning_threads', 'meta_sim_samples']:
            if ( not (options[option] is None) ):
                print ("Updating Meta option: ", option, " = ", options[option])
                hyperSettings_[option] = options[option]
        
        result = tuneHyperParameters(simsettingsFileName=simSettings_['configFile'], simSettings=simSettings_, hyperSettings=hyperSettings_)
        
        root_data_dir = getRootDataDirectory(simSettings_)+"/"
        
        ### Create a tar file of all the sim data
        print ("hyperSettings_['param_to_tune']", hyperSettings_['param_to_tune'])
        print ("hyperSettings_['param_to_tune']", makeNiceName(hyperSettings_['param_to_tune']))
        tarFileName = (root_data_dir + simSettings_['data_folder'] + "/_" + makeNiceName(hyperSettings_['param_to_tune']) +'.tar.gz_') ## gmail doesn't like compressed files....so change the file name ending..
        # tarFileName = (simSettings_['agent_name']+simSettings_['data_folder']+hyperSettings_['param_to_tune']+'.tar.gz')
        dataTar = tarfile.open(tarFileName, mode='w:gz')
        for meta_result in result['meta_sim_result']:
            print (meta_result)
            for simsettings_tmp in meta_result['settings_files']:
                addDataToTarBall(dataTar, simsettings_tmp)
        polt_settings_files = []    
        for hyperSetFile in result['hyper_param_settings_files']:
            print("adding ", hyperSetFile, " to tar file")
            addDataToTarBall(dataTar, simsettings_tmp, fileName=hyperSetFile)
            polt_settings_files.append(hyperSetFile)
        
        figure_file_name = root_data_dir + simSettings_['data_folder'] + "/_" + makeNiceName(hyperSettings_['param_to_tune']) + '_'
        
        print("root_data_dir: ", root_data_dir)
        pictureFileName=None
        try:
            plotMetaDataSimulation(root_data_dir, simSettings_, polt_settings_files, folder=figure_file_name)
            ## Add pictures to tar file
            addPicturesToTarBall(dataTar, simSettings_)
            pictureFileName=[figure_file_name + "Training_curves.png", 
                             figure_file_name + "Training_curves_discounted_error.png"]
        except Exception as e:
            dataTar.close()
            print("Error plotting data there my not be a DISPLAY available.")
            print("Error: ", e)
        dataTar.close()
        
        ## Send an email so I know this has completed
        ## This prints too much data
        result["meta_sim_result"] = None
        contents_ = json.dumps(hyperSettings_, indent=4, sort_keys=True) + "\n" + json.dumps(result, indent=4, sort_keys=True)
        if ( ('testing' in hyperSettings_ and (hyperSettings_['testing']))):
            print("Not simulating, this is a testing run:")
            testing_ = True
        else:
            testing_ = False 
        sendEmail(subject="Simulation complete: " + result['sim_time'], 
                  contents=contents_, hyperSettings=hyperSettings_, simSettings=sys.argv[1], 
                  dataFile=tarFileName, testing=testing_, 
                  pictureFile=pictureFileName)
        