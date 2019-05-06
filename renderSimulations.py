import json
import sys
from util.simOptions import getOptions
import subprocess
from tuneHyperParameters import tuneHyperParameters
import copy
from ModelEvaluation import _modelEvaluation
from util.Multiprocessing import MyPool as ProcessingPool
from util.utils import split

# def runModelEvaluation(settings_file_name, settings=None, runLastModel=False, render=True, exp=None):
	
	### create new process and run the process

if __name__ == "__main__":
	
	
	options = getOptions(sys.argv)
	options = vars(options)
	file = open(options['configFile'])
	simulationsSettings_ = json.load(file)
	file.close()
	
	sim_data = []
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
			hyperSettings_["meta_sim_samples"] = 2
			hyperSettings_["meta_sim_threads"] = 2
			hyperSettings_["tuning_threads"] = 2
			
			result = tuneHyperParameters(simsettingsFileName=simConfigFile, 
										simSettings=simSettings_, hyperSettings=hyperSettings_)
			
			### Render simulation result
			for result in result["meta_sim_result"]:
				
				for file_ in result["sim_file_locations"]:
					print ("")
					try:
						print('file: ' + str(file_))
						file = open(file_)
						simSettings_ = json.load(file)
						file.close()
						simSettings_["save_video_to_file"] = "eval_movie2.mp4"
						simSettings_["visualize_expected_value"] = False
						simSettings_["eval_epochs"] = 2
						simSettings_["shouldRender"] = True
						sim_data.append((sys.argv[1], copy.deepcopy(simSettings_)))
					# modelEvaluation(sys.argv[1], runLastModel=False, settings=simSettings_, render=True)
					except:
						print ("Simulation config missing.")

	numThreads = simulationsSettings_["render_threads"]
	p = ProcessingPool(numThreads, maxtasksperchild=1)
	
	sim_datas = split(sim_data, int(len(sim_data)/numThreads)+1)
	results = []
	for sim_datas_ in sim_datas:
	
		result = p.map(_modelEvaluation, sim_datas_)
		results.extend(result)
	
	print (result)
