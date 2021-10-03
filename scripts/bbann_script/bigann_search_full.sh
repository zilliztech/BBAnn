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
# RESULT_OUTPUT
RESULT_OUTPUT_PREFIX=/data/answers/BBANN-BIGANN-32-500-128-3/
TRUTH_SET_FILE=/data/10M_test/datasets/bigann-10M-gt
# NPROBE
NPROBE_LIST=(50 100 200 300 400 500 600 700 800 900 1000)
# REFINE_NPROBE
# K
K1=128
# METRIC_TYPE
PAGE_PER_BLOCK=3
# ================================================================================
# ===========================End Of Parameters====================================
# ================================================================================

echo "Please run this script with root permission."
echo "Copy this file as well as analyze_stat.py to release/ folder, where project is compiled."

echo "Data Type: uint8 for BIGANN"
echo "Index Folder: " $INDEX_PATH
echo "QUERY_FILE: " $QUERY_FILE
echo "Result Answer Path Prefix: " $RESULT_OUTPUT_PREFIX
echo "TRUTH_SET_FILE: " $TRUTH_SET_FILE
echo "Top K: " $K
echo "K1 (cluster number): " $K1
echo "Metric Type: " $METRIC_TYPE
echo "Page per block: " $PAGE_PER_BLOCK

echo "Result Answer Output Folder: " $RESULT_OUTPUT_PREFIX
mkdir $RESULT_OUTPUT_PREFIX # new a folder for answers

if [ "$?" = "1" ]; then
  echo "Fail to mkdir this folder: " $RESULT_OUTPUT_PREFIX
  echo "Please check this folder!"
  echo "ABORT SEARCHING INDEX!"
else
  for NPROBE in "${NPROBE_LIST[@]}";
  do
    echo "========================================================"
    echo "nprobe: " $NPROBE
    let REFINE_NPROBE=5*NPROBE
    echo "refine nprobe: " $REFINE_NPROBE

    RESULT_OUTPUT=${RESULT_OUTPUT_PREFIX}${NPROBE}_${REFINE_NPROBE}.answer
    echo "Result Answer Path: " $RESULT_OUTPUT
    LOG_FILE=${RESULT_OUTPUT_PREFIX}${NPROBE}_${REFINE_NPROBE}.log
    echo "LOG_FILE: " $LOG_FILE

    echo "Cleaning the OS page cache. Run this if you want a real QPS. If you only want a recall, it is not necessary to kill page cache!"
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
    pkill -f "search_bbann"
    time ./search_bbann $DATA_TYPE $INDEX_PATH $QUERY_FILE $RESULT_OUTPUT $TRUTH_SET_FILE $NPROBE $REFINE_NPROBE $K $K1 $METRIC_TYPE $PAGE_PER_BLOCK | sudo tee $LOG_FILE &
    pid=$!
    pidstat -rud -h -t -p $pid 1 > $LOG_FILE.stat
    wait $pid
    python3 analyze_stat.py $LOG_FILE.stat > $LOG_FILE.max.stat
  done
fi

