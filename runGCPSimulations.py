import json
import sys
from util.simOptions import getOptions
import subprocess

if __name__ == "__main__":
	
	
	options = getOptions(sys.argv)
	options = vars(options)
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
	
	for simConfigFile in simSettings_["metaExps"]:
		## now loop through the above array
		for metaConfig in simSettings_["simConfigs"]:
			arg = ( 
				"pushd /home/glen/playground/RL-Framework;" 
		 		"python3 tuneHyperParameters.py --config=${simConfigFile}" 
		 		" --metaConfig=" + str(metaConfig) + 
		 		" --meta_sim_samples=" + str(simSettings_["meta_sim_samples"]) +
		 		" --meta_sim_threads=" + str(simSettings_["meta_sim_threads"]) +
			 	" --tuning_threads=" + str(simSettings_["tuning_threads"]) +
			 	" --num_rounds=" + str(simSettings_["rounds"]) + " --plot=false --on_policy=fast" 
			 	" --save_experience_memory=continual --continue_training=last "
			 	" --saving_update_freq_num_rounds=1 -p 6 --rollouts=12 --simulation_timeout=1200" 
			 	" --email_log_data_periodically=true " + str(simSettings_["opts"]))
			output=' | tee -a $BORGY_JOB_ID.out'
		### GPU training
	# 	 	arg="source ~/tensorflow/bin/activate; pushd /home/glen/playground/RL-Framework; python3 tuneHyperParameters.py --config=${simConfigFile} --metaConfig=${metaConfig} --meta_sim_samples=3 --meta_sim_threads=3 --tuning_threads=2 --num_rounds=${rounds} --plot=false --on_policy=fast --shouldRender=false --save_experience_memory=continual --continue_training=last --saving_update_freq_num_rounds=1 -p 4 --rollouts=16 --simulation_timeout=1200 --email_log_data_periodically=true --save_video_to_file=eval_movie2.mp4 --visualize_expected_value=false --force_sim_net_to_cpu=true ${opts}"
			arg= arg + " " + output
			command=("submit --restartable --cpu=24 --mem=64 --max-run-time-secs=100000" 
			" -w /home/glen -v /mnt/home/glen:/home/glen --image=images.borgy.elementai.lan/glen:latest2" 
			" -e TERRAINRL_PATH=/home/glen/playground/TerrainRL/ -e RLSIMENV_PATH=/home/glen/playground/RLSimulationEnvironments" 
			" -e HOME=/home/glen -- /bin/bash -c ") + arg
			print ("")
			print("command: " + command)
			# eval $command
			# "${command[@]}"
			cmd = ['borgy'] + command.split(" ")
			# proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			
			# o, e = proc.communicate()
			
			# print('Output: ' + o.decode('ascii'))
			# print('Error: '  + e.decode('ascii'))
			# print('code: ' + str(proc.returncode))


