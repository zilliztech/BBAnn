#!/bin/bash
# To build the pybind11 integration, run the following:

sudo python3 setup.py install

python3 tests/bbann_build_test.py --type float --data_path /data/random-xs/random10000/data_10000_20 --save_path /tmp/dat/ 