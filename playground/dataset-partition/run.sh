python3 build_index.py --partition_type kmeans --dataset_name sift1M --n_cluster 256
python3 build_index.py --partition_type graph_partition --dataset_name sift1M --n_cluster 256
python3 query.py --filename sift1M-graph_partition-knn-k_30 --dataset_name sift1M
python3 query.py --filename sift1M-kmeans --dataset_name sift1M
