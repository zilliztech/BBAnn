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

To compile the debug mode, run
```shell
$ mkdir -p debug
$ cd debug
$ cmake ..
$ make -j
```

To compile the release mode, run
```shell
$ mkdir -p release
$ cd release
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make -j
```

### Current Approach: Block-based index

**Please Build With the Script**: 
```
$ cp scripts/bbann_script/build.sh release/
$ cp scripts/bbann_script/analyze_stat.py release/
$ chmod +x release/build.sh
$ # !!!You have to modify build necessary parameters!!!
$ sudo screen ./release/build.sh
```

[Basic Build Script: build.sh](scripts/bbann_script/build.sh)

#### Build Log Reader Scripts

[Build Log Reader Script: build_log_extract.sh](scripts/bbann_script/build_log_extract.sh)

```
$ cp scripts/bbann_script/build_log_extract.sh [/index_path/log/]
$ cd [/index_path/log/]
$ chmod +x build_log_extract.sh
$ sudo ./build_log_extract.sh
$ cat result.csv
```

#### Search Scripts

**Please Search With the Script**:
```
$ cp scripts/bbann_script/search.sh release/
$ cp scripts/bbann_script/analyze_stat.py release/
$ chmod +x release/search.sh
$ # !!!You have to modify search necessary parameters!!!
$ # !!!You have to make sure the log & answer folder are there!!!
$ sudo ./release/search.sh
```

```
$ cp scripts/bbann_script/search_full.sh release/
$ cp scripts/bbann_script/analyze_stat.py release/
$ chmod +x release/search_full.sh
$ # !!!You have to modify search necessary parameters!!!
$ # !!!You have to make sure the log & answer folder are there!!!
$ sudo ./release/search_full.sh
```

[Basic Search Script with one-nprobe-searching: search.sh](scripts/bbann_script/search.sh)

[Full Search Script with multiple-nprobe-searching: search_full.sh](scripts/bbann_script/search_full.sh)

[Full Search Script for Dataset BIGANN: bigann_search_full.sh](scripts/bbann_script/bigann_search_full.sh)

[Full Search Script for Dataset BIGANN: msspacev_search_full.sh](scripts/bbann_script/msspacev_search_full.sh)

#### Search Log Reader Scripts

[Search Log Reader Script: search_log_extract.sh](scripts/bbann_script/search_log_extract.sh)

```
$ cp scripts/bbann_script/search_log_extract.sh [/answer/index-answer-path/]
$ cp scripts/bbann_script/rewrite.py [/answer/index-answer-path/]
$ cd [/answer/index-answer-path/]
$ chmod +x search_log_extract.sh
$ sudo ./search_log_extract.sh
$ cat result.csv
```

#### Build Index without script

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

#### Search Index without script

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
