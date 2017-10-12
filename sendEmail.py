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
def sendEmail(subject, contents, testing=False):
    # Create a text/plain message
    msg = MIMEText(contents)
    
    hostname = socket.getfqdn()
    
    print("Hostname: ", hostname)
    
    username = getpass.getuser()
    print ("username: ", username)
    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = subject
    # msg['From'] = 'csguestn@dyn.cs.sfu.ca'
    msg['From'] = username
    msg['To'] = 'gberseth@cs.ubc.ca'
    
    if ( testing ):
        return
    
    # Send the message via our own SMTP server.
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()
    


if __name__ == '__main__':
    
    if (len(sys.argv) > 2):
        sendEmail("Testing", "Nothing", testing=False)
    else:
        sendEmail("Testing", "Nothing", testing=True)