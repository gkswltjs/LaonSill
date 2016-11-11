#!/bin/bash
if [ "$#" -ge 3 ]; then
    echo "Usage: build.sh dop [Debug|Release]"
    exit 0
elif [ "$#" -eq 0 ]; then
    echo "Usage: build.sh dop [Debug|Release]"
    exit 0
elif [ "$#" -eq 2 ]; then
    if [[ "$2" == "Debug" ]]; then
        buildDebug=1
        buildRelease=0
    elif [[ "$2" == "Release" ]]; then
        buildDebug=0
        buildRelease=1
    else
        echo "Usage: build.sh dop [Debug|Release]"
        exit 0
    fi
else
    buildDebug=1
    buildRelease=1
fi

dop=$1

cd $NN_HOME

if [ ! -d bin ]; then
    mkdir bin
fi

echo "[generate parameter]"
cd src/param
./genParam.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: build stopped"
    exit -1
fi
cd ../..

echo "[generate hotcode]"
cd src/log
./genHotCode.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: build stopped"
    exit -1
fi
echo "[copy tools related hotcode]"
cp decodeHotLog.py ../../bin/.
cp hotCodeDef.json ../../bin/.
cd ../..

echo "[generate performance]"
cd src/perf
./genPerf.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: build stopped"
    exit -1
fi
cd ../..

if [ "$buildDebug" -eq 1 ]; then
    echo "[build Debug (server)]"
    cd Debug
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    cp NeuralNetworksAndDeepLearning ../bin/SoooaServer
    cd ..

    echo "[build Debug (client)]"
    cd DebugClient
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    cp NeuralNetworksAndDeepLearningClient ../bin/SoooaClient
    cd ..
fi

if [ "$buildRelease" -eq 1 ]; then
    echo "[build Release (server)]"
    cd Release
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    cp NeuralNetworksAndDeepLearning ../bin/SoooaServer
    cd ..

    echo "[build Release (client)]"
    cd ReleaseClient
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    cp NeuralNetworksAndDeepLearningClient ../bin/SoooaClient
    cd ..
fi
