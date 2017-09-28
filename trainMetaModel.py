

from trainModel import trainModelParallel
import sys
import json

def trainMetaModel(settingsFileName, samples=10):
    
    file = open(sys.argv[1])
    settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings)))
    file.close()
    data_name = settings['data_folder']
    for i in range(10):
        settings['data_folder'] = data_name + "_" + str(i) 
        trainModelParallel(settingsFileName, settings)

if (__name__ == "__main__"):
    """
        python trainMetaModel.py <sim_settings_file> <num_samples>
        Example:
        python trainMetaModel.py settings/navGame/PPO_5D.json 10
    """
    
    if (len(sys.argv) == 1):
        print("Please incluse sim settings file")
        print("python trainMetaModel.py <sim_settings_file> <num_samples>")
        sys.exit()
    elif (len(sys.argv) == 2):
        print("Please incluse sim settings file")
        print("python trainMetaModel.py <sim_settings_file> <num_samples>")
        sys.exit()
    elif (len(sys.argv) == 3):
        trainMetaModel(sys.argv[1], samples=int(sys.argv[2]))
    else:
        print("Please specify arguments properly, ")
        print(sys.argv)
        print("python trainMetaModel.py <sim_settings_file> <num_samples>")