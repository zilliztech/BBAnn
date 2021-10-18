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

python3 install.py --algorithm bbann

set -e
pushd ../benchmark
cp ../python/bbann.py benchmark/algorithms/bbann.py
# rm -rf results/random-range-xs/*
python3 run.py --definitions ../python/bbann-algo.yaml --dataset random-range-xs --algorithm bbann # --force --rebuild
python3 plot.py --definitions ../python/bbann-algo.yaml --dataset  random-range-xs --recompute
popd
