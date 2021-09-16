# NIPS Competition

## Prerequisites
* CMake >= 3.10
* gcc >= 6.1
* AIO

To install these dependencies:

```shell
$ chmod +x install.sh
$ sudo ./install.sh
```

## Get Started

To compile the release mode, run
```shell
$ mkdir -p release
$ cd release
$ cmake .. -D CMAKE_BUILD_TYPE=Release
$ make -j
```

### Build Index

To build, under the `build` directory, run `build_bigann` with the following arguments
 1. data type(string): float or uint8 or int8
 2. raw data binary file(string): file name of raw data, include file path and file name
 3. index path(string): a string end with '/' denotes the directory that where the index related file locates
 4. M(int): parameter for hnsw
 5. efConstruction(int): parameter for hnsw
 6. PQ.M(int): the number of codebook for each vector
 7. PQ.nbits(int): the number of sub-cluster for PQ codebook
 8. metric type(string): metric type
 9. K1(int): number of centroids of the first round kmeans
 10. bucket threshold(int): the threshold of spliting the cluster
 11. PQ type(string): PQ or PQRes
 
``` shell
$ mkdir -p index_path
$ ./build_bigann [uint8|int8|float] [raw_data_path] [index_path] [HNSW_M] [HNSW_efConstruction] [PQ_M] [PQ_nbits] [IP|L2] [K1] [bucket_threshold] [PQ|PQRes]
```

### Search Query

To search, run `search_bigann` with the following arguments
args:
1. data type(string): float or uint8 or int8
2. index path(string): a string end with '/' denotes the directory that where the index related file locates
3. query data file(string): binary file of query data
4. answer file(string): file name to store answer
5. ground_truth file(string): file name where ground_truth stores
6. nprobe(int): number of buckets to query
7. refine nprobe(int): number of buckets to be candidate
8. topk(int): number of answers for each query
9. ef(int): number of answers for refine
10. PQ.M(int): the number of codebook for each vector
11. PQ.nbits(int): the number of sub-cluster for PQ codebook
12. K1(int): number of centroids of the first round kmeans
13. metric type(string): metric type
14. quantizer (PQ | PQRes)

``` shell
$ ./search_bigann [uint8|int8|float] [index_path] [query_data_file] [answer_path] [ground_truth_path] [nprobe] [ef] [PQ_M] [PQ_nbits] [K1] [IP|L2] [PQ|PQRes]
```
