#!/bin/bash
# To build the pybind11 integration, run the following:
set -e
mkdir -p ../build
pushd ../build
cmake ..
cd src
make clean
make -j
popd

sudo rm -rf build/
sudo rm -rf /tmp/dat
sudo mkdir /tmp/dat
sudo python3 setup.py install

sudo python3 tests/bbann_build_test.py \
  --type float \
  --data_path /data/random-xs/random10000/data_10000_20 \
  --save_path /tmp/dat/ 

sudo python3 tests/bbann_query_test.py \
   --type float --index_path_prefix /tmp/dat/ \
   --query_path /data/random-xs/random10000/queries_1000_20
echo Succ!
