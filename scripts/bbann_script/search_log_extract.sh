#!/bin/bash

temp="tana_res.txt"
py="form.txt"
result="result.csv"

rm -f $temp
rm -f $py
rm -f $result
DIR=`pwd`
for file in `ls ${DIR}/*.log`;
do
    echo "Analysis Search Log File: " $file >> $temp
    split1=(${file//// })
    file_name=${split1[-1]}
    echo "file name: " ${file_name}
    split2=(${file_name//_/ })
    nprobe=${split2[0]}
    echo "nprobe: " ${nprobe}
    split3=(${split2[-1]//./ })
    efSearch=${split3[0]}
    echo "efSearch: " ${efSearch}
    echo $nprobe" "$efSearch >> $temp

    cat $file | grep "search graph: search graph done." | awk -F' ' '{print $6}' | awk -F'(' '{print $2}' >> $temp
    cat $file | grep "search bigann: heapify answers heaps" | awk -F' ' '{print $6}' | awk -F'(' '{print $2}' >> $temp
    cat $file | grep "search bigann: calculate block position done" | awk -F' ' '{print $7}' | awk -F'(' '{print $2}' >> $temp
    cat $file | grep "search bigann: async io done" | awk -F' ' '{print $6}' | awk -F'(' '{print $2}' >> $temp
    cat $file | grep "search bigann: search bigann totally done" | awk -F' ' '{print $7}' | awk -F'(' '{print $2}' >> $temp
    cat $file | grep "avg recall@10" | awk -F' ' '{print $4}' | awk -F'%' '{print $1}' >> $temp
done

# 6 for search log
python3 rewrite.py 6
cat $py | sort -n > $result
echo "Done. Check table file without any header: " $result
