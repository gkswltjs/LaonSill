#!/bin/bash
mkdir -p LAONSILL_HOME
mkdir -p LAONSILL_HOME/log
mkdir -p LAONSILL_HOME/data
mkdir -p LAONSILL_HOME/param
mkdir -p LAONSILL_HOME/output_images
mkdir -p LAONSILL_HOME/bin
mkdir -p LAONSILL_HOME/lib
mkdir -p LAONSILL_HOME/dev/server
mkdir -p LAONSILL_HOME/dev/client
mkdir -p LAONSILL_HOME/examples
mkdir -p LAONSILL_HOME/pyapi
cp ../template/laonsill.conf.sample LAONSILL_HOME/laonsill.conf
cp ../template/cluster.conf.sample LAONSILL_HOME/cluster.conf
cp ../bin/LaonSillServer LAONSILL_HOME/bin/.
cp ../bin/LaonSillClient LAONSILL_HOME/bin/.
cp ../bin/convert_imageset LAONSILL_HOME/bin/.
cp ../bin/convert_mnist_data LAONSILL_HOME/bin/.
cp ../bin/denormalize_param LAONSILL_HOME/bin/.
cp ../src/3rd_party/nms/libgpu_nms.so LAONSILL_HOME/lib/.
cp -r ../dev/client/* LAONSILL_HOME/dev/client/.
cp -r ../dev/server/* LAONSILL_HOME/dev/server/.
cp -r ../src/pyapi/LaonSill LAONSILL_HOME/pyapi/.

rsync -avz --exclude '*.cpp' --exclude '*.h' --exclude '*.cu' ../src/examples LAONSILL_HOME

hashVal=`git rev-parse --short HEAD`
echo $hashVal >> LAONSILL_HOME/revision.txt
echo `git rev-parse HEAD` >> LAONSILL_HOME/revision.txt
echo `git log -1 --format=%cd` >> LAONSILL_HOME/revision.txt

tar cvfz LaonSill_Rev$hashVal.tar.gz LAONSILL_HOME
rm -rf LAONSILL_HOME
