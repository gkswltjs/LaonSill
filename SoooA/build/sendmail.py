#!/usr/bin/env python

import os
from subprocess import Popen, PIPE

import datetime

###########################################################################################
# Settings
#  - you can modify below things
###########################################################################################

sentence = "12fvdk3rdd4h4dsfnq3fjsme35mfeejjklltxxzabjcda"
dop = "4"
toList = ["monhoney@laonbud.com", "jhkim@laonbud.com", "grandsys@laonbud.com",\
    "twone315@laonbud.com"]

###########################################################################################
# Codes
#  - do not modify below things
###########################################################################################

def send_email(user, pwd, recipient, subject, body):
    import smtplib

    gmail_user = user
    gmail_pwd = pwd
    FROM = user
    TO = recipient if type(recipient) is list else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server_ssl = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server_ssl.ehlo() # optional, called by login()
        server_ssl.login(gmail_user, gmail_pwd)  
        # ssl server doesn't support or need tls, so don't call server_ssl.starttls() 
        server_ssl.sendmail(FROM, TO, message)
        #server_ssl.quit()
        server_ssl.close()
        print 'successfully sent the mail'

    except Exception as e:
        print "failed to send mail"
        print str(e)

def simpleDecode(str):
    result = ""
    charIdx = 0
    for char in str:
        if (charIdx % 3) == 2:
            result += char
        charIdx = charIdx + 1
    return result

try:
    p = Popen(["sh", "%s/build/build.sh" % os.environ['SOOOA_BUILD_PATH'], dop],\
            stdout=PIPE, stderr=PIPE)
    sout, serr = p.communicate()
except Exception as e:
    print str(e)

if p.returncode == 0:
    print 'success'
    buildResult = "%s Build result : success\n\n\n" % datetime.datetime.now()
    buildResult += "============= Build Details =============\n"
    buildResult += sout
    buildResult += "============= Build Details =============\n\n\n"
    isSuccess = True
else:
    print 'fail'
    buildResult = "%s Buildi result : fail\n\n\n" % datetime.datetime.now()

    print "stderr : %s" % serr
    buildResult += "============= ERROR =============\n"
    buildResult += serr
    buildResult += "============= ERROR =============\n\n\n"

    print "stdout : %s" % sout
    buildResult += "============= Build Details =============\n"
    buildResult += sout
    buildResult += "============= Build Details =============\n\n\n"
    isSuccess = False

subject = "daily build(%s)" % datetime.datetime.today().strftime("%Y-%m-%d")
if isSuccess == True:
    subject = subject + ":Success"
else:
    subject = subject + ":Fail"

send_email("developer@laonbud.com", simpleDecode(sentence), toList,\
    subject, buildResult)
