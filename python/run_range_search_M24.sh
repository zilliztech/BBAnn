#!/bin/bash
# To build the pybind11 integration, run the following:
# need to create symbolic link ../benchmark/data at certain SSD.
# then,
# python3 create_dataset.py --datset random-xs

cp utils.py ../benchmark/benchmark/plotting/utils.py


pushd ../
docker build --rm -f Dockerfile.local -t billion-scale-benchmark-bbann .
popd

set -e
pushd ../benchmark
cp ../python/bbann.py benchmark/algorithms/bbann.py
sed -e 's/IDENTIFIER/M24/g' ../python/bbann-algo-1B-template.yaml | sed -e 's/HNSWM/24/g' > ../python/bbann-algo-M24-docker.yaml
rm -rf results/ssnpp-1B*
python3 run.py --definitions ../python/bbann-algo-M24-docker.yaml --dataset ssnpp-1B  --algorithm bbann --runs 1 --count 96237  --force # --rebuild
python3 plot.py --definitions ../python/bbann-algo-M24-docker.yaml --dataset ssnpp-1B --recompute --count 96237
popd
