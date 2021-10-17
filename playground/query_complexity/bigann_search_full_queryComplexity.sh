#!/bin/bash

DATA_TYPE=uint8
METRIC_TYPE=L2
# ================================================================================
# ========================Begin Of Parameters=====================================
# ================================================================================
# DATA_TYPE
INDEX_PATH=/data/index/BBANN-BIGANN-32-500-128-1-32-500-128-1/
QUERY_FILE=/data/datasets/bigann/query.public.10K.u8bin
# RESULT_OUTPUT
RESULT_OUTPUT_PREFIX=/data/answers/query_complexity_continue/
TRUTH_SET_FILE=/data/datasets/bigann/bigann-1B-gt
# NPROBE
#NPROBE_LIST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
NPROBE_LIST=(25 30 35 40 45 50 60 70 80 90 100 150 200 250 300 350 400 450 500)
#NPROBE_LIST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 35 40 45 50 60 70 80 90 100 150 200 250 300 350 400 450 500)
# REFINE_NPROBE
# K
K=1 # TODO: 1, 5, 9
K1=128
# METRIC_TYPE
PAGE_PER_BLOCK=1
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
    time ./search_bbann $DATA_TYPE $INDEX_PATH $QUERY_FILE $RESULT_OUTPUT $TRUTH_SET_FILE $NPROBE $REFINE_NPROBE $K $K1 $METRIC_TYPE $PAGE_PER_BLOCK | sudo tee $LOG_FILE
  done
fi

