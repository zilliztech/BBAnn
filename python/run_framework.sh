#!/bin/bash
# To build the pybind11 integration, run the following:
set -e
pushd ../benchmark
python3 run.py --definitions ../python/bbann-algo.yaml --nodocker --dataset random-xs --algorithm bbann 
python3 plot.py --definitions ../python/bbann-algo.yaml --dataset random-xs
popd
