import pytest
from sendEmail import sendEmail
import json
import tarfile
import junit2htmlreport
from junit2htmlreport.parser import Junit as JunitParser
import sys
import time
import datetime

### Need to go through them in order to avoid issues with Tensorflow state staying around...
tests_ =[
        # 'test_cacla.py',
        # ,'test_ddpg.py'
        # ,'test_fd_models.py'
        # ,'test_learning.py'
        # ,'test_mbrl.py'
        # ,'test_meta_training.py'
        'test_model.py'
        # ,'test_multiAgentNav.py'
        # ,'test_ppo.py'
        # ,'test_ppo_more.py'
        # ,'test_saveandload_fd.py'
        ,'test_saveandload.py'
        ,'test_simulation.py'
        # ,'test_viz_imitation.py',
         ]

def run_tests(metaSettings, test=False):
    if (test):
        print ("test run")
        pytest.main(['tests/test_model.py', '--junitxml=' + metaSettings['j_unit_filename'], '--workers=2', '--tests-per-worker=1'])
    else:
        print ("Starting full run: ")
        for test in tests_:
            print ("Running tests: ", test)
            # command_ = ['tests/' + tests, '--junitxml=' + tests + metaSettings['j_unit_filename'], '--workers', str(metaSettings['tuning_threads']), 
            #              '--tests-per-worker=1', '--show-capture=no', '--timeout_method=thread', '--timeout=30']
            # command_ = ['tests/' + test, '--junitxml=' + metaSettings['j_unit_filename'] + test, '--workers='+ str(metaSettings['tuning_threads']), 
            #               '--tests-per-worker=1']
            command_ = ['tests/' + test, '--junitxml=' + metaSettings['j_unit_filename'] + test, '-n '+ str(metaSettings['tuning_threads'])]
            print ("Command: ", command_)
            pytest.main(command_)
        # pytest.main(['tests/', '--junitxml=' + jUnitFileName, '-n', '4'])
            
if __name__ == '__main__':
    
    hyperSettingsFileName = sys.argv[1] 
    file = open(hyperSettingsFileName)
    hyperSettings_ = json.load(file)
    print ("Settings: " + str(json.dumps(hyperSettings_)))
    file.close()
    
    # pytest.main('-x {0}'.format(argument))
    jUnitFileName = hyperSettings_['j_unit_filename']
    # Or
    t0 = time.time()
    if ( len(sys.argv) == 3 and sys.argv[2] == "Test"):
        run_tests(metaSettings=hyperSettings_, test=True)
    else:
        run_tests(metaSettings=hyperSettings_, test=False)
    t1 = time.time()
    sim_time_ = datetime.timedelta(seconds=(t1-t0))
    print ("Model testing complete in " + str(sim_time_) + " seconds")
    
    tarFileName = ('_sim_data.tar.gz_') ## gmail doesn't like compressed files....so change the file name ending..
    dataTar = tarfile.open(tarFileName, mode='w:gz')
    # addDataToTarBall(dataTar, settings)
    dataTar.close()
    ## Send an email so I know this has completed
    
    contents_ = ""
    for test in tests_: ## Collect test data
        contents_ = contents_ + JunitParser(hyperSettings_['j_unit_filename'] + test).html()
    sendEmail(subject="Simulation complete: " + str(sim_time_), contents="", hyperSettings=hyperSettings_, 
              simSettings="", dataFile=tarFileName,
              pictureFile=None, htmlContent=contents_) 
    
    