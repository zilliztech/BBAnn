#!/bin/bash

DIR=`pwd`
output="tana_res.txt"

for file in `ls ${DIR}/*.log`;
do
    echo "analysis file: " $file >> $output
    split1=(${file//// })
    file_name=${split1[-1]}
    echo "file name: " ${file_name}
    split2=(${file_name//_/ })
    nprobe=${split2[1]}
    echo "nprobe: " ${nprobe}
    split3=(${split2[-1]//./ })
    refine_topk=${split3[0]}
    echo "refine_topk: " ${refine_topk}
    echo $refine_topk >> $output
    #cat $file |  grep "main: load pq codebook done." | awk -F' ' '{print $6}' >> $output
    cat $file |  grep "main: load pq codebook done." | awk -F' ' '{print $6}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "main: load meta done." | awk -F' ' '{print $5}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "main: load hnsw done." | awk -F' ' '{print $5}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "main: load pq centroids done." | awk -F' ' '{print $6}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "search bigann with pipeline: 20-means on query done." | awk -F' ' '{print $9}'  | awk -F'(' '{print $2}' >> $output
    cat $file | grep "search bigann with pipeline: elkan_L2_assign done." | awk -F' ' '{print $7}'  | awk -F'(' '{print $2}' >> $output
    echo "null" >> $output
    cat $file | grep "search bigann with pipeline: start all sync io threads." | awk -F' ' '{print $10}'  | awk -F'(' '{print $2}' >> $output
    cat $file | grep "search bigann with pipeline: search bigann totally done" | awk -F' ' '{print $9}'  | awk -F'(' '{print $2}' >> $output
    cat $file | grep "main:  totally done." | awk -F' ' '{print $4}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "avg recall@10" | awk -F' ' '{print $4}' | awk -F'%' '{print $1}' >> $output
done

