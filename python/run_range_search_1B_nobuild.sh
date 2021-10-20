#!/bin/bash
# To build the pybind11 integration, run the following:
# need to create symbolic link ../benchmark/data at certain SSD.
# then,
# python3 create_dataset.py --datset random-xs

set -e
pushd ../benchmark
cp ../python/bbann.py benchmark/algorithms/bbann.py
rm -rf results/ssnpp-1B*
python3 run.py --definitions ../python/bbann-algo.yaml --nodocker --dataset ssnpp-1B --algorithm bbann --runs 1 # --force --rebuild
python3 plot.py --definitions ../python/bbann-algo.yaml --dataset ssnpp-1B --recompute
popd
