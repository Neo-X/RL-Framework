import pytest
from sendEmail import sendEmail
import json
import tarfile
import junit2htmlreport
from junit2htmlreport.parser import Junit as JunitParser
        
if __name__ == '__main__':
    processes = 4
    # pytest.main('-x {0}'.format(argument))
    jUnitFileName = 'test_output.xml'
    # Or
    # pytest.main(['tests/', '--junitxml=' + jUnitFileName, '-n', '4'])
    pytest.main(['tests/test_model.py', '--junitxml=' + jUnitFileName, '-n', '4'])
    
    hyperSettings_ = {}
    hyperSettings_['from_email_address'] = 'gberseth@cs.ubc.ca'
    hyperSettings_['mail_server_name'] = 'mail.cs.ubc.ca'
    sim_time_ = 0.2
    tarFileName = ('_sim_data.tar.gz_') ## gmail doesn't like compressed files....so change the file name ending..
    dataTar = tarfile.open(tarFileName, mode='w:gz')
    # addDataToTarBall(dataTar, settings)
    dataTar.close()
    ## Send an email so I know this has completed
    contents_ = JunitParser(jUnitFileName).html()
    sendEmail(subject="Simulation complete: " + str(sim_time_), contents=contents_, hyperSettings=hyperSettings_, simSettings="", dataFile=tarFileName,
              pictureFile=None) 