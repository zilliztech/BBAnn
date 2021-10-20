#!/bin/bash
# To build the pybind11 integration, run the following:
# need to create symbolic link ../benchmark/data at certain SSD.
# then,
# python3 create_dataset.py --datset random-xs


set -e
mkdir -p ../build
pushd ../build
cmake ..
cd src/
make TimeRecorder
cd lib
make clean
make -j
popd


sudo rm -rf build/
sudo python3 setup.py install -f

set -e
pushd ../benchmark
cp ../python/bbann.py benchmark/algorithms/bbann.py
rm -rf results/random-range-s/*
python3 run.py --definitions ../python/bbann-algo.yaml --nodocker --dataset random-range-s --algorithm bbann --runs 2 --force # --rebuild 
python3 plot.py --definitions ../python/bbann-algo.yaml --dataset random-range-s --recompute
popd
