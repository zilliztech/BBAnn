python3 analyze_partition.py --partition_type kmeans --dataset_name bigann10K --n_cluster 100 --data_set_path /mnt/Billion-Scale/BIGANN/base.10K.u8bin
python3 analyze_partition.py --partition_type graph_partition --dataset_name bigann10K --n_cluster 100 --data_set_path /mnt/Billion-Scale/BIGANN/base.10K.u8bin

python3 analyze_partition.py --partition_type kmeans --dataset_name bigann100K --n_cluster 1000 --data_set_path /mnt/Billion-Scale/BIGANN/base.100K.u8bin
python3 analyze_partition.py --partition_type graph_partition --dataset_name bigann100K --n_cluster 1000 --data_set_path /mnt/Billion-Scale/BIGANN/base.100K.u8bin


python3 analyze_partition.py --partition_type kmeans --dataset_name bigann1M --n_cluster 10000 --data_set_path /mnt/Billion-Scale/BIGANN/base.1M.u8bin
python3 analyze_partition.py --partition_type graph_partition --dataset_name bigann1M --n_cluster 10000 --data_set_path /mnt/Billion-Scale/BIGANN/base.1M.u8bin