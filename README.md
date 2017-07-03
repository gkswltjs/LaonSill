# Welcome to SoooA framework

SoooA is a C++ based open source software framework. SoooA supports a wide variety of distributed GPUs and offers excellent performance in speed / memory to work well in commercial environments. SoooA is distributed under the Apache 2.0 license and offers paid models such as subscription models with technical support for paid users. Currently, SoooA is developing a framework to run in embedded and mobile environment and will provide functions to analyze / design using GUI.

# Installation
## Required Packages

In addition to this, there may be various libraries requried.

* OPENCV
  * http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html
* CUDA 8.0
  * https://developer.nvidia.com/cuda-downloads
* CUDNN 6.0
  * https://developer.nvidia.com/cudnn
* python 2.x(we recommend above 2.7 version)
  * https://www.python.org/ftp/python/2.7.9/Python-2.7.9.tgz
* Boost Library
  * http://www.boost.org/
* GNUPlot-iostream
  * http://stahlke.org/dan/gnuplot-iostream/
* CImg
  * http://cimg.eu/
* BLAS
  * http://www.netlib.org/blas/

## Setup for Compile
Download the SoooA framework. Then set the top directory path to `$SOOOA_BUILD_PATH`

```
$ cd $HOME
$ git clone https://github.com/laonbud/SoooA.git
$ export SOOOA_BUILD_PATH=$HOME/SoooA
```

Make $SOOOA_HOME in the right place. There are configuration files, log files and so on that are needed for SOOOA framework.

```
$ cd $HOME
$ mkdir SOOOA_HOME
$ export SOOOA_HOME=`pwd`/src
```

Copy the env.sh.eg file located in $SOOOA_BUILD_PATH to the env.sh file. Then set the env.sh file appropriately for your environment. This env.sh file should always be run. It is recommended that you configure env.sh to run in files such as .bashrc, .bash_profile, and so on.

```
$ cd $SOOOA_BUILD_PATH
$ cp env.sh.eg env.sh
$ vi env.sh
[env.sh example]
export SOOOA_HOME="/home/monhoney/SOOOA_HOME"
export INC_PATH_GNUPLOT="/home/monhoney/install/gnuplot-iostream"
export INC_PATH_CIMG="/usr/include"
export SOOOA_SOURCE_PATH="/home/monhoney/SoooA/src"
export SOOOA_BUILD_PATH="/home/monhoney/SoooA"
export LD_LIBRARY_PATH=$SOOOA_SOURCE_PATH/3rd_party/nms:$LD_LIBRARY_PATH
```