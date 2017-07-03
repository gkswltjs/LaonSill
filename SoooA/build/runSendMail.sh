#!/bin/bash

source /opt/rh/devtoolset-3/enable
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
source /home/build/soooa/SoooA/env.sh
unset SSH_ASKPASS
/home/build/soooa/SoooA/build/sendmail.py
