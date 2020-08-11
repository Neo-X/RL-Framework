

import logging
log = logging.getLogger(__file__)
from launchers.config import *

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
    from util.tuneParams import run_sweep
    
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

    settings["exp_name"] = settings["data_folder"]
    settings['settingsFileName'] = sys.argv[1]
    
    sweep_ops={}
    if ( 'tuningConfig' in settings):
        sweep_ops = json.load(open(settings['tuningConfig'], "r"))

    run_sweep(trainModelParallel, sweep_ops=sweep_ops, variant=settings, repeats=settings['meta_sim_samples'],
              meta_threads=settings['meta_sim_threads'])
#     if ( (metaSettings is None)
#         or ((metaSettings is not None) and (not metaSettings['testing'])) ):
#         run_experiment(
#         trainModelParallel,
#         exp_name='test-doodad-easy-launch_rlframe',
# #         mode='local_docker',
#         mode='local',
# #         mode='ec2',
#         variant=settings,
# #         region='us-east-2',
#     )
#     t1 = time.time()
#     sim_time_ = datetime.timedelta(seconds=(t1-t0))
#     print ("Model training complete in " + str(sim_time_) + " seconds")
    
    ### If a metaConfig is supplied email out the results
    print("All Done.")
    sys.exit(0)

if (__name__ == "__main__"):
    main()
