# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText

# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.
# with open(textfile) as fp:
def sendEmail(subject, contents):
    # Create a text/plain message
    msg = MIMEText(contents)
    
    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = subject
    msg['From'] = 'csguestn@dyn.cs.sfu.ca'
    msg['To'] = 'gberseth@cs.ubc.ca'
    
    # Send the message via our own SMTP server.
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()