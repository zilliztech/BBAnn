#!/bin/bash

temp="tana_res.txt"
py="form.txt"
result="result.csv"

rm -f $temp
rm -f $py
rm -f $result
DIR=`pwd`
for file in `ls ${DIR}/*build.log`;
do
    echo "Analysis Build Log File: " $file >> $temp
    cat $file | grep "build bigann: train cluster to get 10 centroids done." | awk -F' ' '{print $10}' | awk -F'(' '{print $2}' >> $temp
    cat $file | grep "build bigann: divide raw data into 10 clusters done" | awk -F' ' '{print $10}' | awk -F'(' '{print $2}' >> $temp
    cat $file | grep "build bigann: conquer each cluster into buckets done" | awk -F' ' '{print $9}' | awk -F'(' '{print $2}' >> $temp
    cat $file | grep "build bigann: build hnsw done." | awk -F' ' '{print $6}' | awk -F'(' '{print $2}' >> $temp
    cat $file | grep "build bigann: gather statistics done" | awk -F' ' '{print $6}'  | awk -F'(' '{print $2}' >> $temp
    cat $file | grep "build bigann: build bigann totally done." | awk -F' ' '{print $7}' | awk -F'(' '{print $2}' >> $temp
    cat $file | grep "#vectors in bucket " | awk -F' ' '{print $5}' >> $temp # AVG
    cat $file | grep "#vectors in bucket " | awk -F' ' '{print $7}' >> $temp # MAX
    cat $file | grep "#vectors in bucket " | awk -F' ' '{print $9}' >> $temp # MIN
    cat $file | grep "Total number of buckets:" | awk -F' ' '{print $5}' | awk -F'%' '{print $1}' >> $temp
    echo "0" >> $temp
    echo "Do it yourself to get Index Total Size at the INDEX FOLDER: du -h /index"
done

awk 'NR!=1' tana_res.txt | awk -v RS= -v OFS=, '{$1 = $1} NR > 1 { print " " } 1' >> $result
echo "Done. Check table file without any header: " $result
