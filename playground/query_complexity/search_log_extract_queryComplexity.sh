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

    cat $file | grep "in range \[100, 100\]:" | awk -F' ' '{print $6}' >> $temp # RECALL 100%
done

# 1 for query complexity
python3 rewrite.py 1
cat $py | sort -n > $result
echo "Done. Check table file without any header: " $result
# CSV Seperator is a SPACE
# HEADER:
# nprobe efSearch 100%RECALL
# 100%RECALL is a counter
