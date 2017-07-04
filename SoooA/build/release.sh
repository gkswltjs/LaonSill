#!/bin/bash
mkdir -p SOOOA_HOME
mkdir -p SOOOA_HOME/log
mkdir -p SOOOA_HOME/data
mkdir -p SOOOA_HOME/network
mkdir -p SOOOA_HOME/stat
mkdir -p SOOOA_HOME/output_images
mkdir -p SOOOA_HOME/bin
mkdir -p SOOOA_HOME/lib
cp ../template/soooa.conf.sample SOOOA_HOME/soooa.conf
cp ../template/cluster.conf.sample SOOOA_HOME/cluster.conf
cp ../bin/SoooaServer SOOOA_HOME/bin/.
cp ../bin/SoooaClient SOOOA_HOME/bin/.
cp ../src/3rd_party/nms/libgpu_nms.so SOOOA_HOME/lib/.

hashVal=`git rev-parse --short HEAD`
echo $hashVal >> SOOOA_HOME/revision.txt
echo `git rev-parse HEAD` >> SOOOA_HOME/revision.txt
echo `git log -1 --format=%cd` >> SOOOA_HOME/revision.txt

tar cvfz SoooA_Rev$hashVal.tar.gz SOOOA_HOME
rm -rf SOOOA_HOME
