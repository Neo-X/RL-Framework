

from ModelEvaluation import modelEvaluation, _modelEvaluation
import sys
import json
import copy
import gc
# from pathos.multiprocessing import ProcessingPool
# from multiprocessing import Pool as ProcessingPool
from util.Multiprocessing import MyPool as ProcessingPool
import multiprocessing
import time
import datetime

from util.SimulationUtil import getDataDirectory, getBaseDataDirectory, getRootDataDirectory, getAgentName
from simulation.LoggingWorker import LoggingWorker

def emailSimData(settings, metaSettings, sim_time_=0, simData={}, exp=None):
    import os
    import tarfile
#     from sendEmail import sendEmail
    from util.SimulationUtil import addDataToTarBall, addPicturesToTarBall
    from tools.PlotMetadataSimulation import plotMetaDataSimulation
    
    directory= getBaseDataDirectory(settings)
    out_file_name=directory+os.path.basename(settings['configFile'])
    root_data_dir = getRootDataDirectory(settings)+"/"
    print ("Saving settings file with data to: ", out_file_name)
    out_file = open(out_file_name, 'w')
    out_file.write(json.dumps(settings, indent=4))
    # file.close()
    out_file.close()
    
    ### Create a tar file of all the sim data
    tarFileName = (root_data_dir + settings['data_folder'] + 'meta_data.tar.gz_') ## gmail doesn't like compressed files....so change the file name ending..
    dataTar = tarfile.open(tarFileName, mode='w:gz')
    for simsettings_tmp in simData['settings_files']:
        print ("root_data dir for result: ", getDataDirectory(simsettings_tmp))
        addDataToTarBall(dataTar, simsettings_tmp)
        
    polt_settings_files = []
    polt_settings_files.append(out_file_name)
    # for hyperSetFile in result['hyper_param_settings_files']:
    #     print("adding ", hyperSetFile, " to tar file")
    #     addDataToTarBall(dataTar, simsettings_tmp, fileName=hyperSetFile)
    #     polt_settings_files.append(hyperSetFile)
        
    figure_file_name = root_data_dir + settings['data_folder'] + "/_" + makeNiceName(metaSettings['param_to_tune']) + '_'
    
    print("root_data_dir: ", root_data_dir)
    pictureFileName=None
    try:
        plotMetaDataSimulation(root_data_dir, settings, polt_settings_files, folder=figure_file_name)
        ## Add pictures to tar file
        addPicturesToTarBall(dataTar, settings)
        pictureFileName=[figure_file_name + "Training_curves.png", 
                         figure_file_name + "Training_curves_discounted_error.png",
                         figure_file_name + "Training_curves_fd.png",
                         figure_file_name + "Training_curves_reward.png"]
    except Exception as e:
        dataTar.close()
        import traceback
        print("Error plotting data there my not be a DISPLAY available.")
        print("Error: ", e)
        print (traceback.format_exc())
    dataTar.close()
    
    
    ## Send an email so I know this has completed
    # simData["settings_files"] = None ## Remove extra info
    simData['sim_time'] = sim_time_
    contents_ = json.dumps(metaSettings, indent=4, sort_keys=True) + "\n" + json.dumps(simData, indent=4, sort_keys=True)
#     try:
#         sendEmail(subject="Simulation Running: " + str(simData['sim_time']), contents=contents_, hyperSettings=metaSettings, 
#                   simSettings=settings['configFile'], dataFile=tarFileName,
#                   pictureFile=pictureFileName)    
#     except Exception as e:
#         print("Error sending email this computer might not be authorized to use the email account.")
#         print("Error: ", e)
#         print (traceback.format_exc())
        
    ### Backup data
    if (("backup_exp_data" in settings)
        and (settings["backup_exp_data"] == True)):
        import subprocess
        try:
            print("Backing up learning data.")
            subprocess.call("./backup_data.sh", shell=True)
        except Exception as e:
            print("Error Backing up data using rsync.")
            print("Error: ", e)
            print (traceback.format_exc())


def makeNiceName(params_to_tune):
    """
        Take the list of parameters to sample over and return a nice
        string that will result in a good filename.
    """
    out = ""
    for p in params_to_tune:
        out = out + "_" + str(p)
    return out

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
    
    
def trainMetaModel(settingsFileName, samples=10, settings=None, numThreads=1, hyperSettings=None):
    import shutil
    import os
    # import tensorflow as tf
    # keras.backend.set_session(tf.Session(graph=tf.Graph()))
    result_data = {}
    result_data['settings_files'] = []
    
    if (settings is None):
        file = open(settingsFileName)
        settings = json.load(file)
        # print ("Settings: " + str(json.dumps(settings)))
        file.close()
    
    from trainModel import trainModelParallel_ as trainModelParallel
        
    print ( "Running ", samples, " simulation(s) over ", numThreads, " Thread(s)")
    settings_original = copy.deepcopy(settings)
    
    
    directory_= getBaseDataDirectory(settings_original)
    if not os.path.exists(directory_):
        os.makedirs(directory_)
    out_file_name=directory_+"settings.json"
    print ("Saving settings file with data to: ", out_file_name)
    out_file = open(out_file_name, 'w')
    out_file.write(json.dumps(settings_original, indent=4))
    # file.close()
    out_file.close()
        
    sim_settings=[]
    sim_settingFileNames=[]
    sim_data = []
    sim_data_files = []
    for i in range(samples):
        settings['data_folder'] = settings_original['data_folder'] + "_" + str(i)
        settings['random_seed'] = int(settings['random_seed']) + ((int(settings['num_available_threads']) + 1) * i)
        ## Change some other settings to reduce memory usage and train faster
        settings['print_level'] = "hyper_train" ### Greatly reduce print statements
        settings["email_log_data_periodically"] = False ### Don't let sub simulations send emails.
        # settings['shouldRender'] = False ### Don't render sub simulations
        settings['visualize_learning'] = False ### Don't create a actively plot learning data
        ### Reduce IO
        settings['saving_update_freq_num_rounds'] = settings_original['saving_update_freq_num_rounds']
        if ("Use_Multi_GPU_Simulation" in settings_original
            and (settings_original["Use_Multi_GPU_Simulation"] == True)):
            settings["GPU_BUS_Index"] = str(i + settings['meta_thread_index']) ### The first one is reserved for rendering
            if ("num_gpus" in settings_original):
                settings["GPU_BUS_Index"] = str(int(settings["GPU_BUS_Index"]) % int(settings_original["num_gpus"]))
            print ("\nGPU bus index: ", settings["GPU_BUS_Index"], "\n")
        
        if ( 'expert_policy_files' in settings):
            for j in range(len(settings['expert_policy_files'])):
                settings['expert_policy_files'][j] = settings_original['expert_policy_files'][j] + "/_" + str(i)
                
        result_data['settings_files'].append(copy.deepcopy(settings))
        
        sim_settings.append(copy.deepcopy(settings))
        sim_settingFileNames.append(settingsFileName)
        settings['settingsFileName']=settingsFileName
        sim_data.append(copy.deepcopy(settings))
        sim_data_files.append(getDataDirectory(settings)+os.path.basename(settingsFileName))
        
        ## Create data directory and copy any desired files to these folders .
        if ( not (hyperSettings is None) ):
            # file = open(hyperSettings)
            hyper_settings = hyperSettings
            # print ("Settings: " + str(json.dumps(settings)))
            # file.close()
            directory= getDataDirectory(settings)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if ('saved_model_path' in hyperSettings):
                print ("Copying fd model: ", hyperSettings['saved_model_path'])
                # shutil.copy2(hyperSettings['saved_model_path'], directory+"forward_dynamics_"+"_Best_pretrain.pkl" )
                shutil.copy2(hyperSettings['saved_model_path'], directory+getAgentName()+"_Best.pkl" )
            if ( 'saved_model_folder' in hyperSettings):
                ### Copy models from other metamodel simulation
                ### Purposefully not copying the "Best" model but the last instead
                shutil.copy2(hyperSettings['saved_model_folder']+"/_" + str(i)+'/'+settings['model_type']+'/'+getAgentName()+".pkl", directory+getAgentName()+"_Best.pkl" )
                

    if (("email_log_data_periodically" in settings_original)
        and (settings_original["email_log_data_periodically"] == True)):
        loggingWorkerQueue = multiprocessing.Queue(1)
        loggingWorker = LoggingWorker(settings_original, 
                                      emailSimData,
                                       loggingWorkerQueue,
                                       simData=result_data)
        loggingWorker.start()        
    p = ProcessingPool(numThreads, maxtasksperchild=1)
    t0 = time.time()
    # print ("hyperSettings: ", hyper_settings)
    if ( (hyperSettings is not None) and ('testing' in hyper_settings and (hyper_settings['testing']))):
        print("Not simulating, this is a testing run:")
    else:
        print (sim_settingFileNames)
        print (sim_data)
        result = p.map(trainModelParallel, [(x, y) for x, y in zip(sim_settingFileNames, sim_data)])
        if ("save_video_to_file" in settings):
            print ("Creating videos of final policies results")
            # p.map(_modelEvaluation, sim_data)
            # loggingWorkerQueue.put("perform_logging")
            
    t1 = time.time()
    print ("Meta model training complete in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds")
    # print (result)
    result_data['sim_time'] = "Meta model training complete in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds"
    result_data['raw_sim_time_in_seconds'] = t1-t0
    result_data['Number_of_simulations_sampled'] = samples
    result_data['Number_of_threads_used'] = numThreads
    result_data['sim_file_locations'] = sim_data_files
    
    if (("email_log_data_periodically" in settings_original)
            and (settings_original["email_log_data_periodically"] == True)):
        loggingWorkerQueue.put(False)
        loggingWorker.join()
        
    return result_data
    # trainModelParallel(settingsFileName, copy.deepcopy(settings))
        
    
    
def trainMetaModel_(args):
    """
        python trainMetaModel.py <hyper_settings_file> <sim_settings_file> <num_samples> <num_threads> <saved_fd_model_path>
        Example:
        python trainMetaModel.py settings/navGame/PPO_5D.json 10
    """
    import json
    import os
    import sys
    args = sys.argv
    
    from util.simOptions import getOptions
    
    options = getOptions(sys.argv)
    options = vars(options)
        
    if ((options['configFile'] == None) 
        or (options['metaConfigFile'] == None)
        or (options['meta_sim_samples'] == None)
        or (options['meta_sim_threads'] == None)):
        print("Please include sim settings file: ", len(args))
        print("python trainMetaModel.py --config=<sim_settings_file> --metaConfig=<hyper_settings_file> --meta_sim_samples=<num_samples> --meta_sim_threads<num_threads>")
        sys.exit()
    else:
        file = open(options['configFile'])
        simSettings_ = json.load(file)
        file.close()
        
        # simSettings_['configFile'] = options['configFile']
        simSettings_['data_folder'] = simSettings_['data_folder'] + "/"
        
        for option in options:
            if ( not (options[option] is None) ):
                print ("Updating option: ", option, " = ", options[option])
                simSettings_[option] = options[option]
                try:
                    simSettings_[option] = json.loads(simSettings_[option])
                except Exception as e:
                    pass # dataTar.close()
                if ( options[option] == 'true'): 
                    simSettings_[option] = True
                elif ( options[option] == 'false'):
                    simSettings_[option] = False
            # settings['num_available_threads'] = options['num_available_threads']
        
        # print ("Settings: " + str(json.dumps(settings, indent=4)))
        hyperSettings_ = None
        if ( 'metaConfigFile' in simSettings_ and (simSettings_['metaConfigFile'] is not None)):
            ### Import meta settings
            file = open(simSettings_['metaConfigFile'])
            hyperSettings_ = json.load(file)
            file.close()
        
        
        simSettings_['meta_thread_index'] = 0
        
        if ( len(args) == 6 ):
            hyperSettings_['saved_model_path'] = args[5]
            result = trainMetaModel(args[1], samples=int(args[3]), settings=copy.deepcopy(simSettings_), numThreads=int(args[4]), hyperSettings=hyperSettings_)
        else:
            result = trainMetaModel(args[1], samples=int(simSettings_['meta_sim_samples']), settings=copy.deepcopy(simSettings_), numThreads=int(simSettings_['meta_sim_threads']), hyperSettings=hyperSettings_)

        if not ("disable_final_emailing" in simSettings_ and simSettings_["disable_final_emailing"]):
            simSettings_.pop("experiment_logging", None)
            emailSimData(simSettings_, hyperSettings_, sim_time_=result['sim_time'], simData=result)

if (__name__ == "__main__"):
        

    trainMetaModel_(sys.argv)
    
    