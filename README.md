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

## Compile
### Using NVIDIA NSIGHT
#### Auto-code generation

SoooA framework supports a variety of automatic code generation via scripts. You must run the scripts before compiling

```
$ cd $SOOOA_BUILD_PATH/src/param
$ ./genParam.py
$ cd $SOOOA_BUILD_PATH/src/prop
$ ./genEnum.py
$ ./genLayerPropList.py
$ ./genNetworkProp.py
$ cd $SOOOA_BUILD_PATH/src/log
$ ./genHotCode.py
$ cd $SOOOA_BUILD_PATH/src/perf
$ ./genPerf.py
```

#### Compile

This project started with NSIGHT. You can compile directly with the NSIGHT by pressing build button. Please refer to the link below for the usage NSIGHT.

https://developer.nvidia.com/nsight-eclipse-edition

### Using build script

We have created out own build scripts that are independent of NSIGHT builds.

#### Prepare buildDef.json

In order to compile, you have to configure it according to your environment. We provide the buildDef.json file to accomplish that goal. Create buildDef.json from the template file(buildDef.json.eg). And then modify the buildDef.json file to suit your environment.

```
cd $SOOOA_BUILD_PATH/build
cp buildDef.json.eg buildDef.json
vi buildDef.json
[buildDef.json]
{
    "LIBS" :
        [
            "cudart", "opencv_core", "cublas", "cudnn", "boost_system", "boost_filesystem",
            "opencv_highgui", "opencv_imgproc", "opencv_features2d", "z", "boost_iostreams",
            "X11", "gpu_nms", "lmdb"
        ],
    "LIBDIRS" :
        [
            "../src/3rd_party/nms"
        ],
    "ARCH" : "compute_60",
    "CODE" : "sm_60"
}

#### Run build script

You can run this script to compile. Note that the first argument of this script represents the degree of parallelism. Put the number of CPU core as an argument.

```
$ cd $SOOOA_BUILD_PATH\build
$ ./build_only.sh 4
```

If a new file is added or there is a missing configuration in the buildDef.json file and you need to create the Makefile again, run the script below:

```
$ ./cleanBuildGen.sh
```

## Setup for running SoooA framework
### Prepare soooa.conf

You need to prepare the `soooa.conf` file under `$SOOOA_HOME`. A `soooa.conf` file defines the settings needed to run SoooA framework. Please refer to `$SOOOA_BUILD_PATH/template/soooa.conf.sample` file for basic format. See the `$SOOOA_BUILD_PATH/src/param/paramDef.json` file for a description of each configuration parameter.

```
[soooa.conf example]
SESS_COUNT=5
GPU_COUNT=1
JOB_CONSUMER_COUNT=6
NETWORK_SAVE_DIR=/home/monhoney/SOOOA_HOME/network
STATFILE_OUTPUT_DIR=/home/monhoney/SOOOA_HOME/stat
IMAGEUTIL_SAVE_DIR=/home/monhoney/SOOOA_HOME/output_images
COLDLOG_DIR=/home/monhoney/SOOOA_HOME/log
HOTLOG_DIR=/home/monhoney/SOOOA_HOME/log
SYSLOG_DIR=/home/monhoney/SOOOA_HOME/log
COLDLOG_LEVEL=0
BASE_DATA_DIR=/data
```

### Prepare cluster.conf

You need to prepare the cluster.conf file under `$SOOOA_HOME`. The cluster.conf file defines GPU settings needed to run SoooA framework. Please refer to `$SOOOA_BUILD_PATH/template/cluster.conf.sample` file for basic format. The configuration file has a list value for the keyword node. One list consists of a tuple with five values. A description of each tuple value is given below:

* 1st value indicates the node ID
* 2nd value indicates server address
* 3rd value indicates server port number
* 4th value indicates GPU device ID
* 5th value indicates GPU memory size(byte)

```
{
    "node" : 
    [
         [0, "127.0.0.1", 13001, 0, 8388608000]
    ]
}
```

## Run SoooA

After compilation, binaries can be found in the following locations:

### Using NVIDIA NSIGHT

* $SOOOA_BUILD_PATH/Debug/SoooaServer
* $SOOOA_BUILD_PATH/Release/SoooaServer
* $SOOOA_BUILD_PATH/DebugClient/SoooaClient
* $SOOOA_BUILD_PATH/ReleaseClient/SoooaClient

### Using build script

* $SOOOA_BUILD_PATH/bin/SoooaServerDebug
* $SOOOA_BUILD_PATH/bin/SoooaServer
* $SOOOA_BUILD_PATH/bin/SoooaClientDebug
* $SOOOA_BUILD_PATH/bin/SoooaClient

### show SoooA version information

Check SoooA version. If you have been working so far, you are ready to run SoooA.

```
$ ./SoooaServer -v
```

# Tutorial