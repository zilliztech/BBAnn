#!/bin/bash

DATA_TYPE=uint8
K=10
METRIC_TYPE=L2
# ================================================================================
# ========================Begin Of Parameters=====================================
# ================================================================================
# DATA_TYPE
INDEX_PATH=/data/10M_test/indexes
QUERY_FILE=/data/10M_test/datasets/query.public.10K.u8bin
RESULT_OUTPUT=/data/answers/a.answer
TRUTH_SET_FILE=/data/10M_test/datasets/bigann-10M-gt
NPROBE=50
REFINE_NPROBE=250
# K
K1=128
# METRIC_TYPE
PAGE_PER_BLOCK=3

LOG_FILE=/data/0922bbann_search.log
# ================================================================================
# ===========================End Of Parameters====================================
# ================================================================================

echo "Please run this script with root permission."
echo "Copy this file as well as analyze_stat.py to release/ folder, where project is compiled."

if [ "$DATA_TYPE" = uint8 ]; then
  echo "Data Type: uint8 for BIGANN, Facebook SimSearchNet++"
elif [ "$DATA_TYPE" = int8 ]; then
  echo "Data Type: int8 for Microsoft SPACEV"
elif [ "$DATA_TYPE" = float ]; then
  echo "float: float for Microsoft Turing-ANNS, Yandex DEEP, Yandex Text-to-Image"
fi

echo "Index Folder: " $INDEX_PATH
echo "QUERY_FILE: " $QUERY_FILE
echo "Result Answer Path: " $RESULT_OUTPUT
echo "TRUTH_SET_FILE: " $TRUTH_SET_FILE
echo "nprobe: " $NPROBE
echo "refine nprobe: " $REFINE_NPROBE
echo "Top K: " $K
echo "K1 (cluster number): " $K1
echo "Metric Type: " $METRIC_TYPE
echo "Page per block: " $PAGE_PER_BLOCK

echo "LOG_FILE: " $LOG_FILE

echo "Cleaning the OS page cache. Run this if you want a real QPS. If you only want a recall, it is not necessary to kill page cache!"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
pkill -f "search_bbann"
time ./search_bbann $DATA_TYPE $INDEX_PATH $QUERY_FILE $RESULT_OUTPUT $TRUTH_SET_FILE $NPROBE $REFINE_NPROBE $K $K1 $METRIC_TYPE $PAGE_PER_BLOCK | sudo tee $LOG_FILE &
pid=$!
pidstat -rud -h -t -p $pid 1 > $LOG_FILE.stat
wait $pid
python3 analyze_stat.py $LOG_FILE.stat > $LOG_FILE.max.stat
