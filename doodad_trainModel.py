

import logging
log = logging.getLogger(__file__)

def main():
    
    """
        python trainModel.py <sim_settings_file>
        Example:
        python trainModel.py settings/navGame/PPO_5D.json 
    """
    import sys
    import json
    from util.simOptions import getOptions
    from trainModel import trainModelParallel
    from doodad.easy_launch.python_function import run_experiment
    
    options = getOptions(sys.argv)
    options = vars(options)
    file = open(options['configFile'])
    settings = json.load(file)
    file.close()
    
    for option in options:
        if ( not (options[option] is None) ):
            log.info("Updating option: {}={} ".format(option, options[option]))
            settings[option] = options[option]
            try:
                settings[option] = json.loads(settings[option])
            except Exception as e:
                pass # dataTar.close()
            if ( options[option] == 'true'):
                settings[option] = True
            elif ( options[option] == 'false'):
                settings[option] = False
    metaSettings = None

    # Tag_FullObserve_SLAC_mini.json: false
    if ( 'metaConfigFile' in settings and (settings['metaConfigFile'] is not None)):
        ### Import meta settings
        file = open(settings['metaConfigFile'])
        metaSettings = json.load(file)
        file.close()

    # Tag_FullObserve_SLAC_mini.json: false
    if 'checkpoint_vid+rounds' in settings:
        # Tag_FullObserve_SLAC_mini.json: false
        if 'save_video_to_file' in settings:
            log.error('\nerror: checkpoint_vid_rounds set but save_video_to_file is unset. Exiting.')        
            sys.exit()
        # Tag_FullObserve_SLAC_mini.json: false            
        elif 'saving_update_freq_num_rounds' not in settings or settings['saving_update_freq_num_rounds'] > settings['checkpoint_vid_rounds']:
            log.warning('saving_update_freq_num_rounds > checkpoint_vid_rounds. Updating saving_update_freq_num_rounds to checkpoing_vid_rounds')
            settings['saving_update_freq_num_rounds'] = settings['checkpoint_vid_rounds']
        else:
            log.warning("Unhandled else statement!")

#     t0 = time.time()
    simData = []
    if ( (metaSettings is None)
        or ((metaSettings is not None) and (not metaSettings['testing'])) ):
#         simData = trainModelParallel((sys.argv[1], settings))
        settings['settingsFileName'] = sys.argv[1]
        run_experiment(
        trainModelParallel,
        exp_name='test-doodad-easy-launch_rlframe',
#         mode='local_docker',
        mode='local',
#         mode='ec2',
        variant=settings,
#         region='us-east-2',
    )
#     t1 = time.time()
#     sim_time_ = datetime.timedelta(seconds=(t1-t0))
#     print ("Model training complete in " + str(sim_time_) + " seconds")
    print ("simData", simData)
    
    ### If a metaConfig is supplied email out the results
    if ( (metaSettings is not None) ):
        settings["email_log_data_periodically"] = True
        settings.pop('save_video_to_file', None)
        settings.pop("experiment_logging", None)
        collectEmailData(settings, metaSettings, sim_time_, simData)

    print("All Done.")
    sys.exit(0)

if (__name__ == "__main__"):
    main()
