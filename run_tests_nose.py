from sendEmail import sendEmail
import json
import tarfile
import junit2htmlreport
from junit2htmlreport.parser import Junit as JunitParser
import sys
import time
import datetime
import nose

def run_tests(metaSettings, test=False):
    if (test):
        print ("test run")
        nose.run(argv=['--where=./tests/', '--tests=tests.test_model.TestModel', '--processes=2', '--with-xunitmp', '--xunitmp-file='+ jUnitFileName, '--process-restartworker', '--process-timeout=600'])
    else:
        print ("Starting full run: ")
        nose.run(argv=['--where=./tests/', '--processes='+ str(metaSettings['tuning_threads']), '--with-xunitmp', '--xunitmp-file='+ jUnitFileName, '--process-restartworker', '--process-timeout=600'])
        # pytest.main(['tests/', '--junitxml=' + jUnitFileName, '-n', '4'])
    print ("Done tests")
           
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
    contents_ = JunitParser(hyperSettings_['j_unit_filename']).html()
    sendEmail(subject="Simulation complete: " + str(sim_time_), contents="", hyperSettings=hyperSettings_, 
              simSettings="", dataFile=tarFileName,
              pictureFile=None, htmlContent=contents_) 
    
    