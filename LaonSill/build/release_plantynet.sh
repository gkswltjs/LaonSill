#!/bin/bash
mkdir -p LAONSILL_HOME
mkdir -p LAONSILL_HOME/bin
mkdir -p LAONSILL_HOME/data/sdf
mkdir -p LAONSILL_HOME/data/image
mkdir -p LAONSILL_HOME/dev/client/lib
mkdir -p LAONSILL_HOME/dev/server
mkdir -p LAONSILL_HOME/log
mkdir -p LAONSILL_HOME/network_def
mkdir -p LAONSILL_HOME/param
mkdir -p LAONSILL_HOME/output
mkdir -p LAONSILL_HOME/lib
mkdir -p LAONSILL_HOME/pyapi

cp ../template/laonsill.conf.sample LAONSILL_HOME/laonsill.conf
cp ../template/cluster.conf.sample LAONSILL_HOME/cluster.conf

cp ../bin/LaonSillServer LAONSILL_HOME/bin/.
cp ../scripts/sdf_convert/convert_imageset.py LAONSILL_HOME/bin/.

cp ../src/3rd_party/nms/libgpu_nms.so LAONSILL_HOME/lib/.

cp -r ../dev/client/* LAONSILL_HOME/dev/client/.
cd ../dev/client/lib
ln -sf libLaonSillClient.so.1.0.1 libLaonSillClient.so
cd ../../../build

cp ../src/examples/VGG16/vgg16_plantynet_*.json LAONSILL_HOME/network_def/.

cp -r ../dev/server/* LAONSILL_HOME/dev/server/.
cp -r ../src/pyapi/LaonSill LAONSILL_HOME/pyapi/.


hashVal=`git rev-parse --short HEAD`
echo $hashVal >> LAONSILL_HOME/revision.txt
echo `git rev-parse HEAD` >> LAONSILL_HOME/revision.txt
echo `git log -1 --format=%cd` >> LAONSILL_HOME/revision.txt

tar cvfz LaonSill_Rev$hashVal.tar.gz LAONSILL_HOME
rm -rf LAONSILL_HOME
