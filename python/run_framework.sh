#!/bin/bash
# To build the pybind11 integration, run the following:
# need to create symbolic link ../benchmark/data at certain SSD.
# then,
# python3 create_dataset.py --datset random-xs

ALGORITHM=bbann
set -e
pushd ../benchmark
cp ../python/bbann.py benchmark/algorithms/bbann.py
rm -rf results/random-xs/*
python3 run.py --definitions ../python/bbann-algo.yaml --nodocker --dataset text2image-10M --algorithm $ALGORITHM --force --rebuild
python3 plot.py --definitions ../python/bbann-algo.yaml --dataset text2image-10M --recompute
popd
