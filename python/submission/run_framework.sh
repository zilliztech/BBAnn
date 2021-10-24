#!/bin/bash
# To build the pybind11 integration, run the following:
# need to create symbolic link ../benchmark/data at certain SSD.
# then,
# python3 create_dataset.py --datset random-xs

ALGORITHM=bbann
DATASET=msspacev-10M
set -e

git clone --single-branch --branch t2/bbann https://github.com/PwzXxm/big-ann-benchmarks.git

ARGS="--dataset $DATASET"

if [[ "$DATASET" =~ .*"ssnpp".* ]]; then
    ARGS="${ARGS} --count 96237"
fi

pushd big-ann-benchmarks
python3 create_dataset.py --dataset $DATASET

python install.py --algorithm $ALGORITHM
python3 run.py $ARGS --timeout 345600 --runs 1 --algorithm $ALGORITHM # --force --rebuild
python3 plot.py $ARGS --recompute
popd
