#NIPS Competition

## Prerequisites
* CMake >= 3.10
* gcc >= 6.1
* AIO

To install these dependencies in **Ubuntu**:

```shell
$ chmod +x install.sh
$ sudo ./install.sh
```

## Get Started

To compile the release mode, run
```shell
$ mkdir -p release
$ cd release
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make -j
```

### Current Approach: Block-based index

Build: 
```
$ cp scripts/bbann_script/build.sh release/
$ chmod +x release/build.sh
$ # You have to modify build necessary parameters
$ sudo screen ./release/build.sh
```

TODO(jigao): recheck search's script

### Build Index

To build, under the `release` directory, run `build_bbann` with the following arguments

1. data type(string): [uint8|int8|float]
2. data file(string): file path of raw data binary file
3. index path(string): a string end with '/' denotes the directory that where the index related file locates
4. hnsw.M(int): M parameter for hnsw
5. hnsw.efConstruction(int): parameter for hnsw
6. metric type(string): [IP|L2]
7. K1(int): cluster number, number of centroids of the first round kmeans
8. page per block(unsigned long): the number of pages in a block

Use a script:

``` shell
$ mkdir -p index_path
$ ./build_bbann [uint8|int8|float] [data_path] [index_path] [HNSW_M] [HNSW_efConstruction] [IP|L2] [K1] [page_per_block]
```

### Search Index

To search, under the `release` directory, run `search_bbann` with the following arguments
args:
1. data type(string): [uint8|int8|float]
2. index path(string): a string end with '/' denotes the directory that where the index related file locates
3. query data file(string): binary file of query data
4. answer file(string): file name to store answer
5. ground_truth file(string): file name where ground_truth stores
6. nprobe(int): number of buckets to query
7. refine nprobe(int): hnsw.efSearch, number of buckets to be candidates
8. topk(int): number of answers for each query
9. K1(int): number of centroids of the first round kmeans
10. metric type(string): [IP|L2]
11. page per block(unsigned long): the number of pages in a block

``` shell
$ ./search_bbann [uint8|int8|float] [index_path] [query_data_file] [answer_file] [ground_truth_path] [nprobe] [ef] [topK] [K1] [IP|L2] [page_per_block]
```

---
---
---

### Previous Approach: `build_bigann` & `search_bigann`

**Please use script in `scripts/test_scripts`**

#### Build Index

To build, under the `release` directory, run `build_bigann` with the following arguments
 1. data type(string): [uint8|int8|float]
 2. data file(string): file path of raw data binary file
 3. index path(string): a string end with '/' denotes the directory that where the index related file locates
 4. hnsw.M(int): M parameter for hnsw
 5. hnsw.efConstruction(int): parameter for hnsw
 6. PQ.M(int): the number of codebook for each vector
 7. PQ.nbits(int): the number of sub-cluster for PQ codebook
 8. metric type(string): [IP|L2]
 9. K1(int): cluster number, number of centroids of the first round kmeans
 10. bucket threshold(int): the threshold of splitting the cluster
 11. PQ quantizer type(string): [PQ|PQRes]
 
Use a script:

``` shell
$ mkdir -p index_path
$ ./build_bigann [uint8|int8|float] [data_path] [index_path] [HNSW_M] [HNSW_efConstruction] [PQ_M] [PQ_nbits] [IP|L2] [K1] [bucket_threshold] [PQ|PQRes]
```

#### Search Query

To search, under the `release` directory, run `search_bigann` with the following arguments
args:
1. data type(string): [uint8|int8|float]
2. index path(string): a string end with '/' denotes the directory that where the index related file locates
3. query data file(string): binary file of query data
4. answer file(string): file name to store answer
5. ground_truth file(string): file name where ground_truth stores
6. nprobe(int): number of buckets to query
7. refine nprobe(int): hnsw.efSearch, number of buckets to be candidates
8. topk(int): number of answers for each query
9. ef(int): refine_topk, number of answers for refine
10. PQ.M(int): the number of codebook for each vector
11. PQ.nbits(int): the number of sub-cluster for PQ codebook
12. K1(int): number of centroids of the first round kmeans
13. metric type(string): [IP|L2]
14. PQ quantizer type(string): [PQ|PQRes]

``` shell
$ ./search_bigann [uint8|int8|float] [index_path] [query_data_file] [answer_file] [ground_truth_path] [nprobe] [ef] [PQ_M] [PQ_nbits] [K1] [IP|L2] [PQ|PQRes]
```
