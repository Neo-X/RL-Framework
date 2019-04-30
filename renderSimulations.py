import json
import sys
from util.simOptions import getOptions
import subprocess
from tuneHyperParameters import tuneHyperParameters
import copy
from ModelEvaluation import modelEvaluation

if __name__ == "__main__":
	
	
	options = getOptions(sys.argv)
	options = vars(options)
	file = open(options['configFile'])
	simulationsSettings_ = json.load(file)
	file.close()
	
	for option in options:
		if ( not (options[option] is None) ):
			print ("Updating option: ", option, " = ", options[option])
			simulationsSettings_[option] = options[option]
			if ( options[option] == 'true'):
				simulationsSettings_[option] = True
			elif ( options[option] == 'false'):
				simulationsSettings_[option] = False
	
	for metaConfig in simulationsSettings_["metaExps"]:
		## now loop through the above array
		for simConfigFile in simulationsSettings_["simConfigs"]:
			metaConfig_ = copy.deepcopy(metaConfig)
			
			file = open(simConfigFile)
			simSettings_ = json.load(file)
			file.close()
			
			file = open(metaConfig)
			hyperSettings_ = json.load(file)
			file.close()
			hyperSettings_["testing"] = True
			
			result = tuneHyperParameters(simsettingsFileName=simConfigFile, 
										simSettings=simSettings_, hyperSettings=hyperSettings_)
			
			### Render simulation result
			for result in result["meta_sim_result"]:
				
				for file_ in result["sim_file_locations"]:
					print ("")
					print('file: ' + str(file_))
					file = open(file_)
					simSettings_ = json.load(file)
					file.close()
					simSettings_["save_video_to_file"] = "eval_movie2.mp4"
					simSettings_["visualize_expected_value"] = False
					modelEvaluation(sys.argv[1], runLastModel=False, settings=simSettings_, render=True)


