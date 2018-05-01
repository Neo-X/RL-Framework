import pytest
from sendEmail import sendEmail
import json
import tarfile
import junit2htmlreport
from junit2htmlreport.parser import Junit as JunitParser
import sys
import time
import datetime
        
if __name__ == '__main__':
    processes = 4
    # pytest.main('-x {0}'.format(argument))
    jUnitFileName = 'test_output.xml'
    # Or
    t0 = time.time()
    if ( len(sys.argv) == 2 and sys.argv[1] == "Test"):
        print ("test run")
        pytest.main(['tests/', '--junitxml=' + jUnitFileName, '--no-print-logs'])
    else:
        print ("Starting full run: ")
        pytest.main(['tests/', '--junitxml=' + jUnitFileName, '-n', '4', '--show-capture=no'])
        # pytest.main(['tests/', '--junitxml=' + jUnitFileName, '-n', '4'])
    t1 = time.time()
    sim_time_ = datetime.timedelta(seconds=(t1-t0))
    print ("Model testing complete in " + str(sim_time_) + " seconds")
    
    hyperSettings_ = {}
    hyperSettings_['from_email_address'] = 'gberseth@cs.ubc.ca'
    hyperSettings_['mail_server_name'] = 'mail.cs.ubc.ca'
    tarFileName = ('_sim_data.tar.gz_') ## gmail doesn't like compressed files....so change the file name ending..
    dataTar = tarfile.open(tarFileName, mode='w:gz')
    # addDataToTarBall(dataTar, settings)
    dataTar.close()
    ## Send an email so I know this has completed
    contents_ = JunitParser(jUnitFileName).html()
    sendEmail(subject="Simulation complete: " + str(sim_time_), contents="", hyperSettings=hyperSettings_, 
              simSettings="", dataFile=tarFileName,
              pictureFile=None, htmlContent=contents_) 
    
    