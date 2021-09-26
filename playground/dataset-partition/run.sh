python3 build_index.py --partition_type kmeans --dataset_name bigann10K --n_cluster 100 --data_set_path /mnt/Billion-Scale/BIGANN/base.10K.u8bin
python3 build_index.py --partition_type graph_partition --dataset_name bigann10K --n_cluster 100 --data_set_path /mnt/Billion-Scale/BIGANN/base.10K.u8bin
#python3 query.py --filename sift1M-graph_partition-knn-k_30 --dataset_name sift1M
#python3 query.py --filename sift1M-kmeans --dataset_name sift1M
