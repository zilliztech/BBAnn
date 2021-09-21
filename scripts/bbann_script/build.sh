#!/bin/bash

# ================================================================================
# ========================Begin Of Parameters=====================================
# ================================================================================
DATA_TYPE=uint8
#DATA_TYPE=int8
#DATA_TYPE=float
DATA_FILE=/mnt/Billion-Scale/BIGANN/base.1B.u8bin
INDEX_PATH=/data/BIGANN
HNSW_M=32
HNSW_EF=500
METRIC_TYPE=L2
#METRIC_TYPE=IP
K1=128
PAGE_PER_BLOCK=3

INDEX_PATH=${INDEX_PATH}-${HNSW_M}-${HNSW_EF}-${K1}-${PAGE_PER_BLOCK}/
LOG_PREFIX=${INDEX_PATH}log/
# ================================================================================
# ===========================End Of Parameters====================================
# ================================================================================

echo "Please run this script with root permission."
echo "Copy this file to release/ folder, where project is compiled."

pkill -f "build_bbann"

if [ "$DATA_TYPE" = uint8 ]; then
  echo "Data Type: uint8 for BIGANN, Facebook SimSearchNet++"
elif [ "$DATA_TYPE" = int8 ]; then
  echo "Data Type: int8 for Microsoft SPACEV"
elif [ "$DATA_TYPE" = float ]; then
  echo "float: float for Microsoft Turing-ANNS, Yandex DEEP, Yandex Text-to-Image"
fi

echo "Data File: " $DATA_FILE

echo "Index Path Folder: " $INDEX_PATH
mkdir $INDEX_PATH # new a folder for indexes

if [ "$?" = "1" ]; then
  echo "Fail to mkdir this folder: " $INDEX_PATH
  echo "Please check this folder!"
  echo "ABORT BUILDING INDEX!"
else
  mkdir $LOG_PREFIX
  LOG_PREFIX=${LOG_PREFIX}build_log

  echo "HNSW's M: " $HNSW_M
  echo "HNSW efConstruction: " $HNSW_EF
  echo "Metric Type: " $METRIC_TYPE
  echo "K1 (cluster number): " $K1
  echo "Page per block: " $PAGE_PER_BLOCK

  # TODO: input sanitizer

  DATE_WITH_TIME=`date "+%Y%m%d_%H%M%S"`
  LOG_PREFIX=${LOG_PREFIX}_${DATE_WITH_TIME}
  LOG_FILE=${LOG_PREFIX}.log
  echo "Log File: " $LOG_FILE

  screen ./build_bbann $DATA_TYPE $DATA_FILE $INDEX_PATH $HNSW_M $HNSW_EF $METRIC_TYPE $K1 $PAGE_PER_BLOCK >$LOG_FILE 2>&1
  pid=$!
  echo "PID: " $pid
  pidstat -rud -h -t -p $pid 5 > ${LOG_PREFIX}.stat
  python3 analyze_stat.py ${LOG_PREFIX}.stat > ${LOG_PREFIX}.max.stat
  wait $pid
fi


