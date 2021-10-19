#pragma once

#include "ivf/kmeans.h"
#include "ivf/balanced_kmeans.h"
#include "ivf/same_size_kmeans.h"
#include <mutex>

template<typename T>
void find_nearest_large_bucket (
        const T *  x, const float * centroids,
        int64_t nx, int64_t k, int64_t dim, int64_t * hassign,
        int64_t * transform_table, std::vector<int64_t>& assign)
{
    std::vector<int64_t> new_assign(nx, -1);
#pragma omp parallel for
    for(int i = 0; i < nx; i++) {
        auto *x_i =  x + i * dim;
        int min_id = 0;
        float min_dist ;
        float dist ;
        if (transform_table[assign[i]] != -1) {
            new_assign[i] = transform_table[assign[i]];
        } else {
            min_dist = L2sqr<const T, const float ,float >(x_i, centroids + transform_table[min_id] * dim, dim);

            for (int j = min_id; j < k; j++) {

                dist = L2sqr<const T, const float, float>(x_i, centroids + transform_table[j] *dim , dim);
                if(dist < min_dist) {
                    min_dist = dist;
                    min_id = j;
                }
            }
            new_assign[i] = min_id;
        }
    }
    assign.assign(new_assign.begin(), new_assign.end());
}

template <typename T>
void merge_clusters(LevelType level, int64_t dim, int64_t nx, int64_t& k, const T* x,
                    std::vector<int64_t>& assign, std::vector<float>& centroids, float avg_len = 0.0) {

    int64_t* hassign = new int64_t [k];
    memset(hassign, 0, sizeof(int64_t) * k);
    for (int i = 0; i < nx; i++) {
        hassign[assign[i]]++;
    }

    int64_t large_bucket_min_limit;
    int64_t small_bucket_max_limit;
    // strategies should be changed according to different scenarios
    if(level == LevelType::FIRST_LEVEL) {

        large_bucket_min_limit = MAX_SAME_SIZE_THRESHOLD;
        small_bucket_max_limit = MAX_SAME_SIZE_THRESHOLD;

    } else {

        large_bucket_min_limit = MAX_SAME_SIZE_THRESHOLD;
        small_bucket_max_limit = MIN_SAME_SIZE_THRESHOLD;

    }

    //find the new k2 and centroids:
    int64_t new_k = 0;
    int64_t large_bucket_num = 0;
    int64_t middle_bucket_num = 0;
    int64_t * transform_table = new int64_t [k]; // old k to new k
    for (int i=0; i < k; i++ ) {
        if(hassign[i] >= large_bucket_min_limit) {
            transform_table[i] = large_bucket_num;
            large_bucket_num++;
        } else {
            transform_table[i] = -1;
        }
    }
    new_k += large_bucket_num;
    for (int i = 0; i < k; i++) {
        if (hassign[i] >= small_bucket_max_limit && transform_table[i] == -1) {
            transform_table[i] = new_k;
            new_k++;
            middle_bucket_num ++;
        }
    }
    if(new_k == k) {

        return ;
    }
    new_k = new_k != 0 ? new_k : 1; // add a bucket for all small bucket

    int64_t * new_hassign = new int64_t [new_k];
    float * new_centroids = new float[dim * new_k];
    for (int i = 0; i < k; i++) {
        if(transform_table[i] != -1) {
            memcpy(new_centroids + transform_table[i] * dim, centroids.data() + i * dim, dim * sizeof(float));
        }
    }
    if (large_bucket_num) {

        find_nearest_large_bucket<T>(x, new_centroids, nx, large_bucket_num, dim, hassign, transform_table, assign);

        compute_centroids<T>(dim, new_k, nx, x, assign.data(), new_hassign, new_centroids, avg_len);

    } else if (middle_bucket_num) {
        find_nearest_large_bucket<T>(x, new_centroids, nx, middle_bucket_num, dim, hassign, transform_table, assign);

        compute_centroids<T>(dim, new_k, nx, x, assign.data(), new_hassign, new_centroids, avg_len);
    } else {

        float * __restrict merge_centroid = new_centroids;
        int64_t merge_centroid_id = 0;
        memset(merge_centroid, 0, sizeof(float) * dim);
#pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            auto * __restrict x_in = x + i * dim;
            if(transform_table[assign[i]] == -1) {
                for (int d = 0; d < dim; d++) {
                    merge_centroid[d] += x_in[d];
                }
                assign[i] = merge_centroid_id;
            } else {
                assign[i] = transform_table[assign[i]];
            }
        }

        if (avg_len != 0.0) {
            float len = avg_len / sqrt(IP<float, float, double>(merge_centroid, merge_centroid, dim));
            for (int64_t j = 0; j < dim; j++){
                merge_centroid[j] *= len;
            }
        } else {
            float norm = 1.0 / nx;
            for (int64_t j = 0; j < dim; j++) {
                merge_centroid[j] *= norm;
            }
        }
    }

    //update meta :
    k = new_k;
    centroids.assign(new_centroids, new_centroids + k * dim);

    delete [] new_centroids;
    delete [] new_hassign;
    delete [] transform_table;
    delete [] hassign;


    return ;
}

struct ClusteringTask{
    ClusteringTask(int64_t o, int64_t n, int64_t l)
    : offset(o), num_elems(n), level(l) {}
    int64_t offset;
    int64_t num_elems;
    int64_t level;
};

template <typename T>
void non_recursive_multilevel_kmeans(uint32_t k1_id, // the index of k1 round k-means
                                     int64_t cluster_size, // num vectors in this cluster
                                     T* data, // buffer to place all vectors
                                     uint32_t* ids, // buffer to place all ids
                                     int64_t round_offset, // the offset of to clustering data in this round
                                     int64_t dim, // the dimension of vector
                                     uint32_t threshold, // determines when to stop recursive clustering
                                     const uint64_t blk_size, // general 4096, determines how many vectors can be placed in a block
                                     uint32_t& blk_num, // output variable, num block output in this round clustering
                                     IOWriter& data_writer, // file writer 1: to output base vectors
                                     IOWriter& centroids_writer, // file writer 2: to output centroid vectors
                                     IOWriter& centroids_id_writer, // file writer 3: to output centroid ids
                                     int64_t centroids_id_start_position, // the start position of all centroids id
                                     int level, // n-th round recursive clustering, start with 0
                                     std::mutex &mutex, // mutex to protect write out centroids
                                     std::vector<ClusteringTask> &output_tasks, // output clustering tasks
                                     bool kmpp = false, // k-means parameter
                                     float avg_len = 0.0, // k-means parameter
                                     int64_t niter = 10, // k-means parameter
                                     int64_t seed = 1234 // k-means parameter
                                     ) {
    // move pointer to current round
    data = data + round_offset * dim;
    ids = ids + round_offset;

    float weight = 0;
    int64_t vector_size = sizeof(T) * dim;
    int64_t id_size = sizeof(uint32_t);

    // Step 0: set the num of cluster in this round clustering

    int64_t k2 = -1; // num cluster in this round clustering
    bool do_same_size_kmeans = (LevelType (level) >= LevelType ::BALANCE_LEVEL) ||
                               +                               (LevelType (level) == LevelType ::THIRTH_LEVEL && cluster_size >= MIN_SAME_SIZE_THRESHOLD && cluster_size <= MAX_SAME_SIZE_THRESHOLD);
    if (do_same_size_kmeans) {
        k2 = std::max((cluster_size + threshold - 1) / threshold, 1L);
    } else {
        k2 = int64_t(sqrt(cluster_size/threshold)) + 1;
        k2 = k2 < MAX_CLUSTER_K2 ? k2 : MAX_CLUSTER_K2;
    }
    assert(k2 != -1);
    std::cout << "step 0: set k2: "
              << "[level " << level << "] "
              << "[cluster_size " << cluster_size << "] "
              << "[k2 " << k2 << "] "
              << "[do same size kmeans " << do_same_size_kmeans << "] "
              << std::endl;

    // Step 1: clustering

    std::vector<float> k2_centroids(k2 * dim, 0.0);
    std::vector<int64_t> cluster_id(cluster_size, -1);

    if (do_same_size_kmeans) {
        //use same size kmeans or graph partition
        k2 = std::max((cluster_size + threshold - 1) / threshold, 1L);
        same_size_kmeans<T>(cluster_size, data, dim, k2, k2_centroids.data(), cluster_id.data(), kmpp, avg_len, niter, seed);
    } else {
        int64_t train_size = cluster_size;
        T* train_data = nullptr;
        if (cluster_size > k2 * K2_MAX_POINTS_PER_CENTROID) {
            train_size = k2 * K2_MAX_POINTS_PER_CENTROID;
            train_data = new T [train_size * dim];
            random_sampling_k2(data, cluster_size, dim, train_size, train_data, seed);
        } else {
            train_data = data;
        }
        kmeans<T>(train_size, train_data, dim, k2, k2_centroids.data(), kmpp, avg_len, niter, seed);
        if(cluster_size > k2 * K2_MAX_POINTS_PER_CENTROID) {
            delete [] train_data;
        }

        // Dynamic balance constraint K-means:
        // balanced_kmeans<T>(cluster_size, data, dim, k2, k2_centroids, weight, kmpp, avg_len, niter, seed);
        std::vector<float> dists(cluster_size, -1);
        if ( weight != 0 && cluster_size <= KMEANS_THRESHOLD ) {
            dynamic_assign<T, float, float>(data, k2_centroids.data(), dim, cluster_size, k2, weight, cluster_id.data(), dists.data());
        } else {
            elkan_L2_assign<T, float, float>(data, k2_centroids.data(), dim, cluster_size, k2, cluster_id.data(), dists.data());
        }

        //dists is useless, so delete first
        std::vector<float>().swap(dists);

        merge_clusters<T>((LevelType)level, dim, cluster_size, k2, data, cluster_id, k2_centroids, avg_len);

        //split_clusters_half(dim, k2, cluster_size, data, nullptr, cluster_id.data(), k2_centroids, avg_len);
    }
    std::cout << "step 1: clustering: " << "cluster centroids be wrote into k2_centroids and cluster_id: "
              << std::endl;


    // Step 2: reorder data by cluster id

    std::vector<int64_t> bucket_pre_size(k2 + 1, 0);
    for (int i=0; i<cluster_size; i++) {
        bucket_pre_size[cluster_id[i]+1]++;
    }
    for (int i=1; i <= k2; i++) {
        bucket_pre_size[i] += bucket_pre_size[i-1];
    }

    // now, elems in bucket_pre_size is prefix sum

    // reorder thr data and ids by their cluster id
    T* x_temp = new T[cluster_size * dim];
    uint32_t* ids_temp = new uint32_t[cluster_size];
    int64_t offest;
    memcpy(x_temp, data, cluster_size * vector_size);
    memcpy(ids_temp, ids, cluster_size * id_size);
    for(int i=0; i < cluster_size; i++) {
        offest = (bucket_pre_size[cluster_id[i]]++);
        ids[offest] = ids_temp[i];
        memcpy(data + offest * dim, x_temp + i * dim, vector_size);
    }
    delete [] x_temp;
    delete [] ids_temp;

    std::cout << "step 2: reorder data by cluster id: "
              << std::endl;

    // Step 3: check all cluster, write out or generate new ClusteringTask

    int64_t bucket_size;
    int64_t bucket_offset;
    int entry_size = vector_size + id_size;

    char* data_blk_buf = new char[blk_size];
    for(int i=0; i < k2; i++) {
        if (i == 0) {
            bucket_size = bucket_pre_size[i];
            bucket_offset = 0;
        } else {
            bucket_size = bucket_pre_size[i] - bucket_pre_size[i - 1];
            bucket_offset = bucket_pre_size[i - 1];
        }
        // std::cout<<"after kmeans : centroids i"<<i<<" has vectors "<<(int)bucket_size<<std::endl;
        if (bucket_size <= threshold) {
            //write a blk to file
            //std::cout << bucket_size<<std::endl;
            memset(data_blk_buf, 0, blk_size);
            *reinterpret_cast<uint32_t*>(data_blk_buf) = bucket_size;
            char* beg_address = data_blk_buf + sizeof(uint32_t);

            for (int j = 0; j < bucket_size; j++) {
                memcpy(beg_address + j * entry_size, data + dim * (bucket_offset + j), vector_size);
                memcpy(beg_address + j * entry_size + vector_size, ids + bucket_offset + j, id_size);
            }

            // need a lock
            {
                std::lock_guard<std::mutex> lock(mutex);

                int64_t current_position = centroids_id_writer.get_position();
                assert(current_position != -1);
                // std::cout << "global_centroids_number: current_position: " << current_position << std::endl;

                // centroid id is uint32_t type
                int64_t blk_num = (current_position - centroids_id_start_position) / sizeof(uint32_t);

                // make sure type cast safety
                assert(blk_num >= 0);

                uint32_t global_id = gen_global_block_id(k1_id, (uint32_t)blk_num);

                // std::cout << "blk_size " << blk_size << std::endl;
                data_writer.write((char *) data_blk_buf, blk_size);
                centroids_writer.write((char *) (k2_centroids.data() + i * dim), sizeof(float) * dim);
                centroids_id_writer.write((char *) (&global_id), sizeof(uint32_t));
            }
        } else {
            output_tasks.emplace_back(ClusteringTask(round_offset + bucket_offset, bucket_size, level + 1));
        }
    }
    delete [] data_blk_buf;
    std::cout << "step 3: write out and generate new ClusteringTask: "
              << "[output_tasks size " << output_tasks.size() << "]"
              << std::endl;
}

template <typename T>
void recursive_kmeans(uint32_t k1_id, int64_t cluster_size, T* data, uint32_t* ids, int64_t dim, uint32_t threshold, const uint64_t blk_size,
                      uint32_t& blk_num, IOWriter& data_writer, IOWriter& centroids_writer, IOWriter& centroids_id_writer, int level,
                      bool kmpp = false, float avg_len = 0.0, int64_t niter = 10, int64_t seed = 1234) {
    float weight = 0;
    int vector_size = sizeof(T) * dim;
    int id_size = sizeof(uint32_t);
    int64_t k2;
    bool do_same_size_kmeans = (LevelType (level) >= LevelType ::BALANCE_LEVEL) ||
            (LevelType (level) == LevelType ::THIRTH_LEVEL && cluster_size >= MIN_SAME_SIZE_THRESHOLD && cluster_size <= MAX_SAME_SIZE_THRESHOLD);
    if (do_same_size_kmeans) {
        k2 = std::max((cluster_size + threshold - 1) / threshold, 1L);
    } else {
        k2 = int64_t(sqrt(cluster_size/threshold)) + 1;
        k2 = k2 < MAX_CLUSTER_K2 ? k2 : MAX_CLUSTER_K2;
    }

    std::cout<< "level" <<level<<" cluster_size "<<cluster_size<< "k2 " << k2 << "do same size kmeans " << do_same_size_kmeans <<std::endl;

    //float* k2_centroids = new float[k2 * dim];
    std::vector<float> k2_centroids(k2 * dim, 0.0);
    std::vector<int64_t> cluster_id(cluster_size, -1);

    if(do_same_size_kmeans) {
        //use same size kmeans or graph partition
        k2 = std::max((cluster_size + threshold - 1) / threshold, 1L);
        same_size_kmeans<T>(cluster_size, data, dim, k2, k2_centroids.data(), cluster_id.data(), kmpp, avg_len, niter, seed);
    } else {
        int64_t train_size = cluster_size;
        T* train_data = nullptr;
        if (cluster_size > k2 * K2_MAX_POINTS_PER_CENTROID) {
            train_size = k2 * K2_MAX_POINTS_PER_CENTROID;
            train_data = new T [train_size * dim];
            random_sampling_k2(data, cluster_size, dim, train_size, train_data, seed);
        } else {
            train_data = data;
        }
        kmeans<T>(train_size, train_data, dim, k2, k2_centroids.data(), kmpp, avg_len, niter, seed);
        if(cluster_size > k2 * K2_MAX_POINTS_PER_CENTROID) {
            delete [] train_data;
        }

        // Dynamic balance constraint K-means:
        // balanced_kmeans<T>(cluster_size, data, dim, k2, k2_centroids, weight, kmpp, avg_len, niter, seed);
        std::vector<float> dists(cluster_size, -1);
        if( weight!=0 && cluster_size <= KMEANS_THRESHOLD ) {
            dynamic_assign<T, float, float>(data, k2_centroids.data(), dim, cluster_size, k2, weight, cluster_id.data(), dists.data());
        } else {
            elkan_L2_assign<T, float, float>(data, k2_centroids.data(), dim, cluster_size, k2, cluster_id.data(), dists.data());
        }

        //dists is useless, so delete first
        std::vector<float>().swap(dists);

        merge_clusters<T>((LevelType)level, dim, cluster_size, k2, data, cluster_id, k2_centroids, avg_len);

        //split_clusters_half(dim, k2, cluster_size, data, nullptr, cluster_id.data(), k2_centroids, avg_len);
    }

    std::vector<int64_t> bucket_pre_size(k2 + 1, 0);

    for (int i=0; i<cluster_size; i++) {
        bucket_pre_size[cluster_id[i]+1]++;
    }
    for (int i=1; i <= k2; i++) {
        bucket_pre_size[i] += bucket_pre_size[i-1];
    }

    //reorder thr data and ids by their cluster id
    T* x_temp = new T[cluster_size * dim];
    uint32_t* ids_temp = new uint32_t[cluster_size];
    int64_t offest;
    memcpy(x_temp, data, cluster_size * vector_size);
    memcpy(ids_temp, ids, cluster_size * id_size);
    for(int i=0; i < cluster_size; i++) {
        offest = (bucket_pre_size[cluster_id[i]]++);
        ids[offest] = ids_temp[i];
        memcpy(data + offest * dim, x_temp + i * dim, vector_size);
    }
    delete [] x_temp;
    delete [] ids_temp;

    int64_t bucket_size;
    int64_t bucket_offest;
    int entry_size = vector_size + id_size;
    uint32_t global_id;

    char* data_blk_buf = new char[blk_size];
    for(int i=0; i < k2; i++) {
        if (i == 0) {
            bucket_size = bucket_pre_size[i];
            bucket_offest = 0;
        } else {
            bucket_size = bucket_pre_size[i] - bucket_pre_size[i - 1];
            bucket_offest = bucket_pre_size[i - 1];
        }
        // std::cout<<"after kmeans : centroids i"<<i<<" has vectors "<<(int)bucket_size<<std::endl;
        if (bucket_size <= threshold) {
            //write a blk to file
            //std::cout << bucket_size<<std::endl;
            memset(data_blk_buf, 0, blk_size);
            *reinterpret_cast<uint32_t*>(data_blk_buf) = bucket_size;
            char* beg_address = data_blk_buf + sizeof(uint32_t);

            for (int j = 0; j < bucket_size; j++) {
                memcpy(beg_address + j * entry_size, data + dim * (bucket_offest + j), vector_size);
                memcpy(beg_address + j * entry_size + vector_size, ids + bucket_offest + j, id_size);
            }
            global_id = gen_global_block_id(k1_id, blk_num);

            data_writer.write((char *) data_blk_buf, blk_size);
            centroids_writer.write((char *) (k2_centroids.data() + i * dim), sizeof(float) * dim);
            centroids_id_writer.write((char *) (&global_id), sizeof(uint32_t));
            blk_num++;

        } else {
            recursive_kmeans(k1_id, bucket_size, data + bucket_offest * dim, ids + bucket_offest, dim, threshold, blk_size,
                             blk_num, data_writer, centroids_writer, centroids_id_writer, level + 1, kmpp, avg_len, niter, seed);
        }
    }
    delete [] data_blk_buf;
}