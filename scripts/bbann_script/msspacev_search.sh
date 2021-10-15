#!/bin/bash

DATA_TYPE=int8
K=10
METRIC_TYPE=L2
# ================================================================================
# ========================Begin Of Parameters=====================================
# ================================================================================
# DATA_TYPE
INDEX_PATH=/data/index/BBANN-msspacev-32-500-128-1-32-500-128-1/
QUERY_FILE=/data/query.i8bin
RESULT_OUTPUT=/data/answers/a.answer
TRUTH_SET_FILE=/data/msspacev-1B-gt
NPROBE=50
REFINE_NPROBE=250
# K
K1=128
# METRIC_TYPE
PAGE_PER_BLOCK=1

LOG_FILE=/data/msspacev_search.log
# ================================================================================
# ===========================End Of Parameters====================================
# ================================================================================

echo "Please run this script with root permission."
echo "Copy this file as well as analyze_stat.py to release/ folder, where project is compiled."

echo "Data Type: int8 for MSSPACE"
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

if [ $NPROBE -eq 50 ]; then
  echo "nprobe=50. Check if your RECALL is ~74.5562% !!!"
elif [ $NPROBE -eq 100 ]; then
  echo "nprobe=100. Check if your RECALL is ~79.7831% !!!"
elif [ $NPROBE -eq 200 ]; then
  echo "nprobe=200. Check if your RECALL is ~84.2939% !!!"
elif [ $NPROBE -eq 300 ]; then
  echo "nprobe=300. Check if your RECALL is ~86.5408% !!!"
else
  echo "unknown nprobe & recall"
fi

