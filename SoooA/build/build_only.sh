#!/bin/bash
clientLibHeaderFiles=(../src/client/ClientAPI.h)
serverLibHeaderFiles=()

if [ "$#" -ge 3 ]; then
    echo "Usage: build_only.sh dop [debug|release|tool|lib]"
    exit 0
elif [ "$#" -eq 0 ]; then
    echo "Usage: build_only.sh dop [debug|release|tool|lib]"
    exit 0
elif [ "$#" -eq 2 ]; then
    if [[ "$2" == "debug" ]]; then
        buildDebug=1
        buildRelease=0
        buildTool=0
        buildLib=0
    elif [[ "$2" == "release" ]]; then
        buildDebug=0
        buildRelease=1
        buildTool=0
        buildLib=0
    elif [[ "$2" == "tool" ]]; then
        buildDebug=0
        buildRelease=0
        buildTool=1
        buildLib=0
    elif [[ "$2" == "lib" ]]; then
        buildDebug=0
        buildRelease=0
        buildTool=0
        buildLib=1
    else
        echo "Usage: build_only.sh dop [debug|release|tool|lib]"
        exit 0
    fi
else
    buildDebug=1
    buildRelease=1
    buildTool=1
    buildLib=1
fi

dop=$1

cd $SOOOA_BUILD_PATH

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

if [ "$buildTool" -eq 1 ]; then
    echo "[build convert imageset tool]"
    cd ToolImageGen
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    cp convert_imageset ../bin/.
    cd ..

    echo "[build convert mnist data tool]"
    cd ToolMnistGen
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    cp convert_mnist_data ../bin/.
    cd ..

    echo "[build denormalize param tool]"
    cd ToolDenormGen
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    cp denormalize_param ../bin/.
    cd ..
fi

if [ "$buildLib" -eq 1 ]; then
    echo "[build client lib]"
    cd ClientLib
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    mkdir -p ../dev/client/lib
    cp lib* ../dev/client/lib/.

    mkdir -p ../dev/client/inc
    for i in ${clientLibHeaderFiles[@]}; do
        cp ${i} ../dev/client/inc/.
    done
    cd ..

    echo "[build server lib]"
    cd ServerLib
    make -j$dop all
    if [ "$?" -ne 0 ]; then
        echo "ERROR: build stopped"
        exit -1
    fi
    mkdir -p ../dev/server/lib
    cp lib* ../dev/server/lib/.

    mkdir -p ../dev/server/inc
    for i in ${serverLibHeaderFiles[@]}; do
        cp ${i} ../dev/server/inc/.
    done
    cd ..
fi
