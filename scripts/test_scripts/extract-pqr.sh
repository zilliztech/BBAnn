#!/bin/bash

DIR=`pwd`
output="tana_res.txt"

for file in `ls ${DIR}/search*`;
do
    echo "analysis file: " $file >> $output
    split1=(${file//// })
    file_name=${split1[-1]}
    echo "file name: " ${file_name}
    split2=(${file_name//-/ })
    nprobe=${split2[1]}
    echo "nprobe: " ${nprobe}
    split3=(${split2[-1]//./ })
    refine_topk=${split3[0]}
    echo "refine_topk: " ${refine_topk}
    echo $nprobe"-"$refine_topk >> $output
    #cat $file |  grep "main: load pq codebook done." | awk -F' ' '{print $6}' >> $output
    cat $file |  grep "main: load pq codebook done." | awk -F' ' '{print $6}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "main: load meta done." | awk -F' ' '{print $5}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "main: load hnsw done." | awk -F' ' '{print $5}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "main: load pq centroids done." | awk -F' ' '{print $6}' | awk -F'(' '{print $2}' >> $output
    #cat $file | grep "search bigann: load query done." | awk -F' ' '{print $6}' >> $output
    #cat $file | grep "earch bigann: knn_2 done" | awk -F' ' '{print $5}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "search bigann: search buckets done." | awk -F' ' '{print $6}' | awk -F'(' '{print $2}' >> $output
    #cat $file | grep "search bigann: pq search done." | awk -F' ' '{print $6}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "search bigann: pq residual search done." | awk -F' ' '{print $7}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "search bigann: refine done" | awk -F' ' '{print $5}' | awk -F'(' '{print $2}' >> $output
    #cat $file | grep "search bigann: write answers done" | awk -F' ' '{print $6}' >> $output
    cat $file | grep "search bigann: search bigann totally done" | awk -F' ' '{print $7}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "main:  totally done." | awk -F' ' '{print $4}' | awk -F'(' '{print $2}' >> $output
    cat $file | grep "avg recall@100" | awk -F' ' '{print $4}' | awk -F'%' '{print $1}' >> $output
    cat $file | grep "total refine vectors:" | awk -F' ' '{print $8}' | awk -F',' '{print $1}' >> $output
    cat $file | grep "total refine vectors:" | awk -F' ' '{print $16}'  | awk -F',' '{print $1}'>> $output
    cat $file | grep "total refine vectors:" | awk -F' ' '{print $12}'  | awk -F',' '{print $1}'>> $output
    cat $file | grep "total refine vectors:" | awk -F' ' '{print $20}'  | awk -F',' '{print $1}'>> $output
    cat $file | grep "total refine vectors:" | awk -F' ' '{print $23}'  | awk -F',' '{print $1}'>> $output
    cat $file | grep "total refine vectors:" | awk -F' ' '{print $4}'  | awk -F',' '{print $1}'>> $output
done

