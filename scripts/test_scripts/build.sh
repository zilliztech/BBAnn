#!/bin/bash

pkill -f "build_bigann"

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
./build_bigann float /mnt/Billion-Scale/Yandex-Text-to-Image/base.1B.fdata /home/zilliz/bigann/test/ 32 500 50 8 IP 128 500 PQRes > build_${DATE_WITH_TIME}.log &

pid=$!
pidstat -rud -h -t -p $pid 5 > build_${DATE_WITH_TIME}.stat

wait $pid
