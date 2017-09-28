

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
    
    trainMetaModel(sys.argv[1], samples=int(sys.argv[2]))