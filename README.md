# Block-based Approximate Nearest Neighbor (BBAnn)
BBAnn is an algorithm optimized for SSD storage. It organizes data so that they are aligned with SSD block size. 

This index is a candidate participating [Billion-Scale Approximate Nearest Neighbor Search Challenge](http://big-ann-benchmarks.com/) Track 2, which searches on a machine with 64GB RAM + 1TB NVMe SSD. 

The source code is mainly located in `include` and `src` folders.
By running scripts under `python` directory, it will create docker image, install python library bound with [pybind11](https://github.com/pybind/pybind11) and then run the benchmark framework.

## Prerequisites
* CMake >= 3.10
* gcc >= 6.1
* AIO
* Docker

## Get Started

``` shell
git clone --recurse-submodules https://github.com/zilliztech/BBAnn.git
cd BBAnn/python

# Run knn search
sudo ./run_framework.sh

# Run range search
sudo ./run_range_search.sh
```

To run a dataset other than `random-xs` and `random-range-xs`, you first need to prepare the dataset

``` shell
cd BBAnn/benchmark
sudo python3 create_dataset.py --dataset [dataset_name]
cd ../python

# Change dataset name
vi run_framework.sh 

sudo ./run_framework.sh
```

The parameters for datasets are located in `python/bbann-algo.yaml`.