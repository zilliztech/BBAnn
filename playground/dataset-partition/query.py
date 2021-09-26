import numpy as np
import json
import bin_io
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='sift10K-kmeans', type=str, help='file name of index')
    parser.add_argument('--dataset_name', default='sift10K', type=str, help='file name of index')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = parse_args()
    basic_index_dir = '/home/zhengbian/nips-competition/dataset-partition/result/%s' % args.filename
    basic_data_dir = '/home/zhengbian/nips-competition/data/%s' % args.dataset_name

    centroid_dir = '%s/centroids.npy' % basic_index_dir
    centroids = np.load(centroid_dir)
    with open('%s/cluster2item.json' % basic_index_dir, 'r') as fp:
        cluster_item_l = json.load(fp)

    gnd, n_query, topk = bin_io.ibin_read('%s/gnd.ibin' % basic_data_dir)
    query, n_query, vec_dim = bin_io.bbin_read('%s/query.bbin' % basic_data_dir)

    n_cluster = len(centroids) # IVF

    total_item_l = np.zeros(shape=n_cluster)   # number of hits in each cluster
    total_recall_l = np.zeros(shape=n_cluster) # recall in each cluster
    for i, tmp_query in enumerate(query, 0):
        distance_l = [np.linalg.norm(tmp_query - _) for _ in centroids]
        distance_sort_l = np.argsort(distance_l)  # Sort distance with each centroids
        tmp_set = []
        tmp_set_len = 0
        tmp_item_l = []
        tmp_recall_l = []

        # Each iteration is one-more nprobe
        for tmp_idx in distance_sort_l:
            tmp_set = np.union1d(tmp_set, cluster_item_l[tmp_idx])
            tmp_set = np.intersect1d(tmp_set, gnd[i])
            tmp_recall = len(tmp_set) / topk
            tmp_set_len += len(cluster_item_l[tmp_idx]) # Add the number of elements in this cluster
            tmp_item_l.append(tmp_set_len)
            tmp_recall_l.append(tmp_recall)
        print(i)

        total_item_l += np.array(tmp_item_l)
        total_recall_l += np.array(tmp_recall_l)

    total_item_l /= n_query
    total_recall_l /= n_query
    print(total_item_l)
    print(total_recall_l)

    item_recall_l = []
    for i in range(len(total_item_l)):
        item_recall_l.append({'n_candidate': total_item_l[i], 'nprobe': i, 'recall': total_recall_l[i]})

    with open('%s/item_recall_curve.json' % basic_index_dir, 'w') as f:
        json.dump(item_recall_l, f)
