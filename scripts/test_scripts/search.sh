#!/bin/bash

# ================================================================================
# ========================Begin Of Parameters=====================================
# ================================================================================
#DATA_TYPE=uint8
#DATA_TYPE=int8
DATA_TYPE=float
INDEX_PATH=/data/BIGANN-32-500-32-8-128-500-PQRes/
QUERY_FILE=/data/query.public.10K.u8bin
RESULT_OUTPUT=/data/answers/
TRUTH_SET_FILE=/data/bigann-1B-gt
NPROBE=50
REFINE_NPROBE=250
# K
EF=50
PQ_M=32
PQ_NBITS=8
K1=128
METRIC_TYPE=L2
#METRIC_TYPE=IP
#PQ_TYPE=PQ
PQ_TYPE=PQRes

LOG_FILE=/data/bigann_search.log
# ================================================================================
# ===========================End Of Parameters====================================
# ================================================================================

echo "Please run this script with root permission."
echo "Copy this file to release/ folder, where project is compiled."

if [ "$DATA_TYPE" = uint8 ]; then
  echo "Data Type: uint8 for BIGANN, Facebook SimSearchNet++"
elif [ "$DATA_TYPE" = int8 ]; then
  echo "Data Type: int8 for Microsoft SPACEV"
elif [ "$DATA_TYPE" = float ]; then
  echo "float: float for Microsoft Turing-ANNS, Yandex DEEP, Yandex Text-to-Image"
fi

echo "Index Folder: " $INDEX_PATH
echo "QUERY_FILE: " $QUERY_FILE
echo "Result Answer Path Folder: " $RESULT_OUTPUT
mkdir $RESULT_OUTPUT

if [ "$?" = "1" ]; then
  echo "Fail to mkdir this folder: " $RESULT_OUTPUT
  echo "Please check this folder!"
  echo "Abort Searching!"
else
  echo "TRUTH_SET_FILE: " $TRUTH_SET_FILE
  echo "nprobe: " $NPROBE
  echo "refine nprobe: " $REFINE_NPROBE
  K=10
  echo "Top K: " $K
  ecbo "EF Refine Topk" $EF
  echo "PQ's M: " $PQ_M
  echo "PQ' Nbits: " $PQ_NBITS
  echo "K1 (cluster number): " $K1
  echo "Metric Type: " $METRIC_TYPE
  echo "Bucket Threshold: " $BUCKET_THRESHOLD
  echo "PQ Quantizer Type: " $PQ_TYPE

  echo "LOG_FILE: " $LOG_FILE

  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
  pkill -f "search_bigann"
	time ./search_bigann $DATA_TYPE $INDEX_PATH $QUERY_FILE $RESULT_OUTPUT $TRUTH_SET_FILE $NPROBE $REFINE_NPROBE $K $EF $PQ_M $PQ_NBITS $K1 $METRIC_TYPE $PQ_TYPE > $LOG_FILE &
  pid=$!
  pidstat -rud -h -t -p $pid 1 > $LOG_FILE.stat
  wait $pid
  python3 analyze_stat.py $LOG_FILE.stat > $LOG_FILE.max.stat
  # TODO: input sanitizer

fi


