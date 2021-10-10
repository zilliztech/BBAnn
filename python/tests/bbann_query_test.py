import time
import argparse
import bbannpy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default=float,
                    help='float or uint8 or int8')
parser.add_argument('--query_path', type=str,
                    help='Path to the input query set of vectors.')

parser.add_argument('--index_path_prefix', type=str,
                    help='Index path prefix.')
parser.add_argument('--K', type=int, default=20, help='k value for recall@K.')
parser.add_argument('--nprobe', type=int, default=2, help='nprobe')
parser.add_argument('--efC', type=int, default=200, help='Parameter for hnsw')
parser.add_argument('--metric', type=str, default='L2',
                    help='Metric type, can be L2 or INNER_PRODUCT')
parser.add_argument('--K1', type=int, default=20,
                    help=' number of centroid of the first round kmeans')
parser.add_argument('--page_per_block', type=int, default=1,
                    help='number of pages in a block')

args = parser.parse_args()

start = time.time()
if 'float' == args.type:
    index = bbannpy.FloatIndex(bbannpy.Metric.L2)


para = bbannpy.BBAnnParameters()

para.metric = para.metric = bbannpy.Metric.L2
para.queryPath = args.query_path
# para.answerFile = args.answer_file
# para.groundTruthFilePath = args.ground_truth_path
para.K = args.K
para.indexPrefixPath = args.index_path_prefix
para.nProbe = args.nprobe
para.hnswefC = args.efC
para.K1 = args.K1
para.blockSize = args.page_per_block * 4096  # pagesize=4096

print("Here")

query_data = bbannpy.VectorFloat()
# ground_truth_ids = bbannpy.VectorUnsigned()
# ground_truth_dists = bbannpy.VectorFloat()

print("Here2")

num_queries, query_dims, query_aligned_dims = bbannpy.read_bin_float(
    para.queryPath, query_data)
print("Here3")

query_data_numpy = np.zeros((num_queries, query_dims), dtype=np.float32)
for i in range(0, num_queries):
    for d in range(0, query_dims):
        query_data_numpy[i, d] = query_data[i * query_aligned_dims + d]
print("Here4")

index = bbannpy.FloatIndex(bbannpy.L2)
print("Here5")

if not index.LoadIndex(para.indexPrefixPath):
    print("error loading index")
print("Load succ")
ids, dists = index.batch_search(query_data_numpy, query_aligned_dims,
                                num_queries, para.K,
                                para)
print(ids[0:10])
print(dists[0:10])
