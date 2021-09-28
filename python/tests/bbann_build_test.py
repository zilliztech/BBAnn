import time
import argparse
import bbannpy

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default=float,
                    help='float or uint8 or int8')
parser.add_argument('--data_path', type=str,
                    help='Path to the input base set of vectors.')
parser.add_argument('--save_path', type=str, help='Path to the built index.')
parser.add_argument('--M', type=int, default=32, help='Parameter for hnsw')
parser.add_argument('--efC', type=int, default=200, help='Parameter for hnsw')
parser.add_argument('--metric', type=str, default='L2',
                    help='Metric type, can be L2 or INNER_PRODUCT')
parser.add_argument('--K1', type=int, default=20,
                    help=' number of centroid of the first round kmeans')
parser.add_argument('--page_per_block', type=int, default=1,
                    help='number of pages in a block')

args = parser.parse_args()

start = time.time()
para =  bbannpy.BBAnnParameters()

para.metric = bbannpy.Metric.L2
para.dataFilePath = args.data_path
para.indexPrefixPath = args.save_path
para.hnswM = args.M
para.hnswefC = args.efC
para.K1 = args.K1
para.blockSize = args.page_per_block * 4096 # pagesize=4096


if 'float' == args.type:
    print("Building float index")
    index = bbannpy.FloatIndex(para.metric)

index.build(para)

end = time.time()

print("Indexing Time: " + str(end - start) + " seconds")
