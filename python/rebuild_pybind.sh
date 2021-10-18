#!/bin/bash
# To build the pybind11 integration, run the following:
set -e
mkdir -p ../build
pushd ../build
cmake ..
cd src/
make TimeRecorder
cd lib
make clean
make BBAnnLib2_s
popd

# sudo pip3 uninstall bbannpy -y
rm -rf build/
sudo python3 setup.py install -f
rm -rf /tmp/dat
mkdir -p /tmp/dat

sudo python3 tests/bbann_build_test.py \
  --type float \
  --data_path /data/random-xs/random10000/data_10000_20 \
  --save_path /tmp/dat/ 

sudo python3 tests/bbann_query_test.py \
   --type float --index_path_prefix /tmp/dat/ \
   --query_path /data/random-xs/random10000/queries_1000_20
echo Succ!
