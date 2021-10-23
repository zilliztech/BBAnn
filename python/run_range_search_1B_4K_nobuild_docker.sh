#!/bin/bash
# To build the pybind11 integration, run the following:
# need to create symbolic link ../benchmark/data at certain SSD.
# then,
# python3 create_dataset.py --datset random-xs


pushd ../
docker build --rm -f Dockerfile.local -t billion-scale-benchmark-bbann .
popd

set -e
pushd ../benchmark
cp ../python/bbann.py benchmark/algorithms/bbann.py
sed -e 's/BLOCKSIZE/4096/g' ../python/bbann-algo-1B-template.yaml  | sed -e 's/INDEXPATH/"\/data\/BBANN-SSNPP-24-500-128-1\/BBANN-SSNPP-24-500-128-1"/g' > ../python/bbann-algo-4K.yaml 
rm -rf results/ssnpp-1B*
python3 run.py --definitions ../python/bbann-algo-4K.yaml --dataset ssnpp-1B  --nodocker --algorithm bbann --runs 1 --count 96237 # --force --rebuild
python3 plot.py --definitions ../python/bbann-algo-4K.yaml --dataset ssnpp-1B --recompute --count 96237
popd
