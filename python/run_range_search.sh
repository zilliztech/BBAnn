#!/bin/bash
# To build the pybind11 integration, run the following:
# need to create symbolic link ../benchmark/data at certain SSD.
# then,
# python3 create_dataset.py --datset random-xs

set -e
mkdir -p ../build
pushd ../build
cmake .. -DCMAKE_BUILD_TYPE=Release
cd src/
make TimeRecorder
cd lib
make clean
make BBAnnLib2_s
popd


sudo rm -rf build/
sudo python3 setup.py install -f

set -e
pushd ../benchmark
cp ../python/bbann.py benchmark/algorithms/bbann.py
rm -rf results/ssnpp-10M
python3 run.py --definitions ../python/bbann-algo.yaml --nodocker --dataset ssnpp-10M --algorithm bbann --runs 1  --force --rebuild
python3 plot.py --definitions ../python/bbann-algo.yaml --dataset ssnpp-10M --recompute
popd
