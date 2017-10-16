# Import smtplib for the actual sending function
import smtplib
import socket
import sys
import getpass

# Import the email modules we'll need
from email.mime.text import MIMEText

# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.
# with open(textfile) as fp:
def sendEmail(subject, contents, settings, simSettings=None, testing=False):
    # Create a text/plain message
    messageBody = contents + "\n" + simSettings
    msg = MIMEText(messageBody)
    
    hostname = socket.getfqdn()
    
    print("Hostname: ", hostname)
    
    username = getpass.getuser()
    print ("username: ", username)
    
    fromEmail = settings['from_email_address']
    print("From email: ", fromEmail)
    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = subject
    # msg['From'] = 'csguestn@dyn.cs.sfu.ca'
    msg['From'] = fromEmail
    msg['To'] = 'gberseth@cs.ubc.ca'
    
    print("Email message: ", msg)
    
    if ( testing ):
        return
    
    # Send the message via our own SMTP server.
    s = smtplib.SMTP(settings['mail_server_name'])
    s.send_message(msg)
    s.quit()
    print ("Email sent.")


if __name__ == '__main__':
    import json
        
    print ("len(sys.argv): ", len(sys.argv))
    if (len(sys.argv) == 2):
        settingsFileName = sys.argv[1] 
        file = open(settingsFileName)
        settings_ = json.load(file)
        print ("Settings: " + str(json.dumps(settings_)))
        file.close()
        
        sendEmail("Testing", "Nothing", settings=settings_, testing=True)
    elif (len(sys.argv) == 3):
        settingsFileName = sys.argv[1] 
        file = open(settingsFileName)
        settings_ = json.load(file)
        print ("Settings: " + str(json.dumps(settings_)))
        file.close()
        
        sendEmail("Testing", "Nothing", settings=settings_, testing=False)
    
    elif (len(sys.argv) == 4):
        settingsFileName = sys.argv[1] 
        file = open(settingsFileName)
        settings_ = json.load(file)
        print ("Settings: " + str(json.dumps(settings_)))
        file.close()
        
        sendEmail("Testing", "Nothing", settings=settings_, simSettings=sys.argv[2], testing=False)
        
    else:
        sendEmail("Testing", "Nothing", testing=True)