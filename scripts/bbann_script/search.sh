#!/bin/bash

# ================================================================================
# ========================Begin Of Parameters=====================================
# ================================================================================
DATA_TYPE=uint8
#DATA_TYPE=int8
#DATA_TYPE=float
INDEX_PATH=/home/bianzheng/LuoJiaRui/data/indices/BigANN10M_gp_version-32-500-128-3/
QUERY_FILE=/home/bianzheng/LuoJiaRui/Dataset/BigANN10M/query.public.10K.u8bin
RESULT_OUTPUT=/home/bianzheng/LuoJiaRui/data/answers/BigANN10M_gp/10021927_result_top10
TRUTH_SET_FILE=/home/bianzheng/LuoJiaRui/Dataset/BigANN10M/bigann-10M-gt
NPROBE=50
REFINE_NPROBE=250
# K
K1=128
METRIC_TYPE=L2
#METRIC_TYPE=IP
PAGE_PER_BLOCK=3

LOG_FILE=/data/0918bbann_search.log
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
K=10
echo "Top K: " $K
echo "K1 (cluster number): " $K1
echo "Metric Type: " $METRIC_TYPE
echo "Page per block: " $PAGE_PER_BLOCK

echo "LOG_FILE: " $LOG_FILE

sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
pkill -f "search_bbann"
time /home/bianzheng/LuoJiaRui/BigANNGP/release/search_bbann $DATA_TYPE $INDEX_PATH $QUERY_FILE $RESULT_OUTPUT $TRUTH_SET_FILE $NPROBE $REFINE_NPROBE $K $K1 $METRIC_TYPE $PAGE_PER_BLOCK | sudo tee $LOG_FILE &
pid=$!
pidstat -rud -h -t -p $pid 1 > $LOG_FILE.stat
wait $pid
python3 analyze_stat.py $LOG_FILE.stat > $LOG_FILE.max.stat
