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

cd $SOOOA_BUILD_PATH

echo "[update to the latest source]"
git pull
if [ "$?" -ne 0 ]; then
    echo "ERROR: git pull failed"
    exit -1
fi

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

echo "[generate enum def]"
cd src/prop
./genEnum.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: build stopped"
    exit -1
fi
cd ../..

echo "[generate layer prop]"
cd src/prop
./genLayerPropList.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: build stopped"
    exit -1
fi
cd ../..

echo "[generate network prop]"
cd src/prop
./genNetworkProp.py
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

echo "[generate auto-makefile]"
cd build
./genMakefile.py
if [ "$?" -ne 0 ]; then
    echo "ERROR: generate auto-makefile failed"
    exit -1
fi
cd ..

if [ "$buildDebug" -eq 1 ]; then
    echo "[build Debug (server)]"
    cd DebugGen
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    cp SoooaServerDebug ../bin/SoooaServerDebug
    cd ..

    echo "[build Debug (client)]"
    cd DebugClientGen
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    cp SoooaClientDebug ../bin/SoooaClientDebug
    cd ..
fi

if [ "$buildRelease" -eq 1 ]; then
    echo "[build Release (server)]"
    cd ReleaseGen
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    cp SoooaServer ../bin/SoooaServer
    cd ..

    echo "[build Release (client)]"
    cd ReleaseClientGen
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    cp SoooaClient ../bin/SoooaClient
    cd ..
fi

echo "[remove build directory]"
rm -rf DebugGen
rm -rf DebugClientGen
rm -rf ReleaseGen
rm -rf ReleaseClientGen

