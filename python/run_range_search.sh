#!/bin/bash
# To build the pybind11 integration, run the following:
# need to create symbolic link ../benchmark/data at certain SSD.
# then,
# python3 create_dataset.py --datset random-xs

set -e
pushd ../benchmark
cp ../python/bbann.py benchmark/algorithms/bbann.py
python3 run.py --definitions ../python/bbann-algo.yaml --nodocker --dataset random-range-xs --algorithm bbann --force --rebuild
python3 plot.py --definitions ../python/bbann-algo.yaml --dataset random-xs --recompute
popd
