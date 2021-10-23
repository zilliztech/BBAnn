#!/bin/bash
# To build the pybind11 integration, run the following:
# need to create symbolic link ../benchmark/data at certain SSD.
# then,
# python3 create_dataset.py --datset random-xs

ALGORITHM=bbann
set -e

pushd ../
docker build --rm -f Dockerfile.local -t billion-scale-benchmark-bbann .
popd

pushd ../benchmark
cp ../python/bbann.py benchmark/algorithms/bbann.py
# rm -rf results/random-xs/*

# cp ../python/Dockerfile.bbann benchmark/install/
# python3 install.py --algorithm bbann

python3 run.py --definitions ../python/bbann-algo.yaml --dataset random-xs --algorithm $ALGORITHM # --force --rebuild
python3 plot.py --definitions ../python/bbann-algo.yaml --dataset random-xs --recompute
popd
