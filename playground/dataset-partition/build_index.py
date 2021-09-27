import argparse
import bin_io
import time
import json
import os
import json
import numpy as np
import sklearn.cluster as cls
import multiprocessing
import faiss
import statistics
import matplotlib.pyplot as plt

graph_knn_k = 30
# kahip_dir = '/home/zhengbian/software/KaHIP'
kahip_dir = ''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cluster', default=256, type=int, help='number of cluseters')
    parser.add_argument('--partition_type', default='kmeans', type=str,
                        help='type of partition method, kmeans or graph_partition')
    parser.add_argument('--dataset_name', default='sift10K', type=str, help='dataset name')
    parser.add_argument('--data_set_path', default='/mnt/Billion-Scale/BIGANN/base.1B.u8bin', type=str, help='data_set_path')
    parser.add_argument('--result_path', default='/home/zilliz/jigao/', type=str, help='result_path')
    parser.add_argument('--kahip_dir', default="/home/zilliz/jigao/KaHIP", type=str, help='kahip_dir')
    parser.add_argument('--query_file_path', default="/mnt/Billion-Scale/BIGANN/query.public.10K.128.u8bin", type=str, help='query_file_path')
    opt = parser.parse_args()
    return opt

def partition_factory(partition_type):
    if partition_type == 'kmeans':
        return kmeans
    elif partition_type == 'graph_partition':
        return graph_partition
    else:
        raise Exception("not support partition type")

def delete_dir_if_exist(dire):
    if os.path.isdir(dire):
        # command = 'sudo rm -rf %s' % dire
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def get_labels(labels, n_cluster):
    n_point_cluster_l = []
    label_map_l = []
    for cluster_i in range(n_cluster):
        base_idx_i = np.argwhere(labels == cluster_i).reshape(-1)
        label_map_l.append(base_idx_i.tolist())
        n_point_cluster_l.append(len(base_idx_i))
    return label_map_l, n_point_cluster_l


def kmeans(base, n_cluster, save_dir):
    ins = cls.KMeans(n_clusters=n_cluster, init='k-means++').fit(base)
    return ins.cluster_centers_, ins.labels_


def graph_partition(base_vector, n_cluster, save_dir):
    def groundtruth(base_vector, query, k):
        tmp_base = base_vector.astype(np.float32)
        tmp_query = query.astype(np.float32)
        base_dim = base_vector.shape[1]
        print("DEBUG base_dim: ", type(base_vector.shape[1]))
        print("DEBUG base_dim: ", base_vector.shape[1])
        print("DEBUG base_dim: ", base_dim)
        index = faiss.IndexFlatL2(base_dim)
        index.add(tmp_base)
        gnd_distance, gnd_idx = index.search(tmp_query, k)
        print("search")
        return gnd_idx

    def build_graph(base_vector, base_base_gnd):
        num_base = len(base_vector)
        index_arr = base_base_gnd
        index_arr = index_arr[:, :] + 1  # kahip need the index start from 1, so +1
        weightless_graph = index_arr.tolist()
        for i in range(len(weightless_graph)):
            weightless_graph[i] = set(weightless_graph[i])

        # print("get the nearest k result")

        for i in range(len(weightless_graph)):
            if (i + 1) in weightless_graph[i]:
                weightless_graph[i].remove((i + 1))
            for n_base_index in weightless_graph[i]:
                if (i + 1) not in weightless_graph[n_base_index - 1]:
                    weightless_graph[n_base_index - 1].add(i + 1)

        res_graph = []
        for i in range(len(weightless_graph)):
            tmp_line = {}
            for num_base in weightless_graph[i]:
                tmp_line[num_base] = 1
            res_graph.append(tmp_line)
        # print("change the rank into graph successfully")
        return res_graph

    def save_graph(graph, save_fname):
        # graph is the 2d array
        vertices = len(graph)
        edges = 0
        for vecs in graph:
            edges += len(vecs)
        assert edges % 2 == 0
        edges = edges / 2

        with open(save_fname, 'w') as f:
            f.write("%d %d 1\n" % (vertices, edges))
            for nearest_index in graph:
                row_index = ""
                for item in nearest_index:
                    row_index += str(item) + " " + str(nearest_index[item]) + " "
                # print(row_index)
                f.write(row_index + '\n')
        print("save graph complete")

    base_base_gnd = groundtruth(base_vector, base_vector, graph_knn_k)
    graph = build_graph(base_vector, base_base_gnd)
    save_fname = '%s/knn.graph' % save_dir
    save_graph(graph, save_fname)
    # graph partition
    kahip_command = 'mpirun -n %d %s/deploy/parhip %s --preconfiguration %s ' \
                    '--save_partition --k %d' % (
                        multiprocessing.cpu_count() // 2, kahip_dir, save_fname,
                        'fastsocial', n_cluster)
    print(kahip_command)
    os.system(kahip_command)
    os.system('mv ./tmppartition.txtp %s/partition.txt' % save_dir)
    label_l = np.loadtxt('%s/partition.txt' % save_dir)

    centroids = []
    vec_dim = len(base_vector[0])

    for cluster_i in range(n_cluster):
        tmp_centroid = np.zeros(shape=vec_dim)
        base_idx_i = np.argwhere(label_l == cluster_i).reshape(-1)
        for tmp in base_idx_i:
            tmp_centroid += base_vector[tmp]
        tmp_centroid = tmp_centroid / len(base_idx_i) if len(base_idx_i) != 0 else tmp_centroid
        centroids.append(tmp_centroid)
    return centroids, label_l


if __name__ == '__main__':
    print("ONLY FOR BIGANN DATASET!")
    args = parse_args()
    print("The partition type: ", args.partition_type)
    print("The dataset name: ", args.dataset_name)
    print("The n_cluster: ", args.n_cluster)
    print("The data_set_path: ", args.data_set_path)
    print("The result_path: ", args.result_path)
    print("The kahip_dir: ", args.kahip_dir)
    print("The query_file_path: ", args.query_file_path)

# TODO: replace varibla with args.varaible

    # basic_dir = '/home/zhengbian/nips-competition/data/%s' % args.dataset_name
    data_set_path = args.data_set_path
    print('The entered data_set_path: ', data_set_path)
    # basic_dir = data_set_path + args.dataset_name

    # save_basic_dir = '/home/zhengbian/nips-competition/dataset-partition/result/%s' % file_name
    result_path = args.result_path
    print('The entered result_path: ', result_path)

    file_name = '%s-%s' % (args.dataset_name, args.partition_type)
    if args.partition_type == 'kmeans':
        pass
    elif args.partition_type == 'graph_partition':
        file_name = '%s-%s-knn-k_%d' % (args.dataset_name, args.partition_type, graph_knn_k)
    save_basic_dir = result_path + file_name
    print('The save_basic_dir: ', save_basic_dir)
    delete_dir_if_exist(save_basic_dir)
    os.mkdir(save_basic_dir)

    kahip_dir = args.kahip_dir
    print('The kahip_dir: ', kahip_dir)

    # base, num_base, dim = bin_io.bbin_read("%s/base.bbin" % basic_dir)
    base_vector, num_vector, dim = bin_io.bbin_read(data_set_path)
    print("base_vector's num_vector: ", num_vector)
    print("base_vector's DIM: ", dim)
    print("base_vector's .shape[1]: ", base_vector.shape[1])

    para_result = {}
    partition_method = partition_factory(args.partition_type)

    print("Start building index.")
    start_time = time.time()
    centroids, labels = partition_method(base_vector, args.n_cluster, save_basic_dir)
    label_map_l, n_point_cluster_l = get_labels(labels, args.n_cluster)
    end_time = time.time()
    para_result['build_index_time'] = end_time - start_time
    para_result['cluster_len'] = n_point_cluster_l
    print("Building index is a success.")

    # Statistics of distribution of elements in clusters.
    # TODO: add to JSON
    print(n_point_cluster_l)
    print("max: ", max(n_point_cluster_l))
    print("min: ", min(n_point_cluster_l))
    print("SUM (should be the same of the number of vector in base file): ", sum(n_point_cluster_l))
    print("avg: ", statistics.mean(n_point_cluster_l))
    print("median: ", statistics.median(n_point_cluster_l))
    plt.hist(n_point_cluster_l)
    plt.title('histogram_' + args.partition_type + "_" + args.dataset_name)
    plt.xlabel('number of vector in a cluster: VALUE RANGE as a BIN')
    plt.ylabel('number of clusters in each bin')
    plt.savefig(result_path + 'histogram_' + args.partition_type + "_" + args.dataset_name + '.png')

    with open('%s/cluster2item.json' % save_basic_dir, 'w') as f:
        json.dump(label_map_l, f)
    with open('%s/index_para_result.json' % save_basic_dir, 'w') as f:
        json.dump(para_result, f)
    np.save('%s/centroids.npy' % save_basic_dir, centroids)
    print("Saving index is a success")


    # TODO: move query.py here

    # TODO: read query file
    query, num_query, query_dim = bin_io.bbin_read(args.query_file_path)
    print("query_vector's num_query: ", num_query)
    print("query_vector's DIM: ", query_dim)
    print("query_vector's .shape[1]: ", query.shape[1])
    top_k = 10
    print("top_k: ", top_k)
    num_vector_per_page = 93 # FIX VALUE WITH 93
    print("num_vector_per_page: ", num_vector_per_page)

    # Build a FLAT index on base vector as float
    print("DEBUG dim: ", dim)
    index = faiss.IndexFlatL2(dim) # TODO: debuggy
    tmp_vector_base = base_vector.astype(np.float32)
    index.add(tmp_vector_base)

    # Search each query in this FLAT index to get the Ground-Truth
    total_item_l = np.zeros(shape = args.n_cluster)   # number of hits in each cluster
    total_page_l = np.zeros(shape = args.n_cluster)   # number of hits in each cluster
    total_recall_l = np.zeros(shape = args.n_cluster) # recall in each cluster
    for i, tmp_query in enumerate(query, 0):
        # Use FLAT to get the Ground-Truth
        tmp_query = query.astype(np.float32)
        gnd_distance, gnd_idx = index.search(tmp_query, top_k)
        # TODO: I only use gnd index, not using the distance

        # nprobe with finding centroids
        distance_l = [np.linalg.norm(tmp_query - _) for _ in centroids]
        distance_sort_l = np.argsort(distance_l)  # Sort distance&index with each centroids
        tmp_set = [] # search result: vector ids
        tmp_set_len = 0
        tmp_page_number = 0
        tmp_item_l = []
        tmp_page_l = []
        tmp_recall_l = []

        # Each iteration is one-more nprobe
        for tmp_idx in distance_sort_l:
            tmp_set = np.union1d(tmp_set, label_map_l[tmp_idx])
            tmp_set = np.intersect1d(tmp_set, gnd_idx[i]) # compare search result with the Ground Truth
            tmp_recall = len(tmp_set) / top_k
            tmp_recall_l.append(tmp_recall)
            tmp_set_len += len(label_map_l[tmp_idx]) # Add the number of elements in this cluster
            tmp_item_l.append(tmp_set_len)
            tmp_page_number += int((len(label_map_l[tmp_idx]) + num_vector_per_page - 1)  / num_vector_per_page)
            tmp_page_l.append(tmp_page_number)
        print(i) # TODO: what is this??? delete this line

        total_item_l += np.array(tmp_item_l)
        total_page_l += np.array(tmp_page_l)
        total_recall_l += np.array(tmp_recall_l)

    total_item_l /= num_query
    total_page_l /= num_query
    total_recall_l /= num_query
    print("AVG number of calculated vectors (distance calculation) per Query: ", total_item_l)
    print("AVG number of Accessed Pages (having ", num_vector_per_page, " vectors)  per Query: ", total_page_l)
    print("AVG RECALL@10 per Query: ", total_recall_l)

    item_recall_l = []
    for i in range(len(total_item_l)):
        item_recall_l.append({'n_candidate': total_item_l[i], 'nprobe': i, 'recall': total_recall_l[i]})
    with open('%s/item_recall_curve.json' % save_basic_dir, 'w') as f:
        json.dump(item_recall_l, f)