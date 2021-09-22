#!/bin/bash

#nprobe=(50 100 200 300 400 500 600 700 800 900 1000)
nprobe=(1000 2000 3000 4000)
lennprobe=${#nprobe[@]}

#refinetopk=(50 100 200 400 800 1000 1600 2000)
refinetopk=(10 20 30 40 50 60 70 80 90 100)

for npb in ${nprobe[@]};
do
    let rnpb=npb*3
    echo "nprobe: " $npb " rnpb: " $rnpb
    echo "--------------------------------------------------------"
    for rt in ${refinetopk[@]};
    do
	echo "refine topk: " $rt
	sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

    pkill -f "search_bigann"

    DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
    F=search-${npb}-${rnpb}-10-${rt}_${DATE_WITH_TIME}
	time ./search_bigann float /home/zilliz/bigann/test/ /mnt/Billion-Scale/Yandex-Text-to-Image/query.public.100K.fbin /home/zilliz/bigann/test/answers/YTI_32_500_50_8_IP_128_500_normalize_${npb}_${rt}_top10_small_rt.answer /mnt/Billion-Scale/Yandex-Text-to-Image/text2image-1B-gt ${npb} ${rnpb} 10 ${rt} 50 8 128 IP PQRes > /home/zilliz/bigann/test/logs/small-rt/${F}.log &

    pid=$!
    pidstat -rud -h -t -p $pid 1 > ${F}.stat
    wait $pid
    python3 analyze_stat.py ${F}.stat > ${F}.max.stat

    echo "--------------------------------------------------------"
    done
done
