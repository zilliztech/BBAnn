#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cassert>


uint32_t page_size = 4096;

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "Usage:" << argv[0]
                  << " data type(float or uint8 or int8)"
                  << " index path"
                  << " cluster number"
                  << " dimension of data"
                  << std::endl;
        return 1;
    }
    std::string index_path(argv[2]);
    int cluster_num = std::stoi(argv[3]);
    if ('/' != *index_path.rbegin())
        index_path += '/';
    int vector_size = std::stoi(argv[4]);
    if (std::string("float") == argv[1]) {
        vector_size *= sizeof(float);
    } else if (std::string("uint8") == argv[1] || std::string("int8") == argv[1]) {
        vector_size *= 1;
    } else {
        std::cout << "wrong data type, must be float or uint8 or int8!" << std::endl;
        return 1;
    }
    int id_size = sizeof(uint32_t);
    int node_size = vector_size + id_size;


    char* dat_buf = new char[page_size];
    char* ids_buf = new char[page_size];
    char* node_buf = new char[page_size];
    memset(dat_buf, 0, page_size);
    memset(ids_buf, 0, page_size);
    memset(node_buf, 0, page_size);

    int nvpp, nipp, nnpp, npv, npi;
    nnpp = page_size / node_size;
    std::vector<std::ifstream> raw_data_file_handlers(cluster_num);
    std::vector<std::ifstream> ids_data_file_handlers(cluster_num);
    std::vector<std::ofstream> data_writer(cluster_num);
    for (auto i = 0; i < cluster_num; i ++) {
        std::string aligned_data_filei = index_path + "cluster-" + "-" + std::to_string(i) + "raw_data" + ".bin";
        std::string aligned_ids_filei  = index_path + "cluster-" + "-" + std::to_string(i) + "global_ids" + ".bin";
        std::string aligned_data_filei2 = index_path + "cluster-" + std::to_string(i) + "-" + "raw_data" + ".bin";
        raw_data_file_handlers[i] = std::ifstream(aligned_data_filei, std::ios::binary);
        ids_data_file_handlers[i] = std::ifstream(aligned_ids_filei , std::ios::binary);
        data_writer[i] = std::ofstream(aligned_data_filei2, std::ios::binary);
        uint32_t clu_size, clu_dim, clu_id_size, clu_id_dim;
        uint32_t ps, check;
        raw_data_file_handlers[i].read(dat_buf, page_size);
        ids_data_file_handlers[i].read(ids_buf, page_size);
        memcpy(&clu_size, dat_buf + 0 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&clu_dim , dat_buf + 1 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&ps      , dat_buf + 2 * sizeof(uint32_t), sizeof(uint32_t));
        assert(ps == page_size);
        memcpy(&nvpp    , dat_buf + 3 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&npv     , dat_buf + 4 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&check   , dat_buf + 5 * sizeof(uint32_t), sizeof(uint32_t));
        assert(check == i);
        memcpy(&clu_id_size, ids_buf + 0 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&clu_id_dim , ids_buf + 1 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&ps         , ids_buf + 2 * sizeof(uint32_t), sizeof(uint32_t));
        assert(ps == page_size);
        memcpy(&nipp    , ids_buf + 3 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&npi     , ids_buf + 4 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&check   , ids_buf + 5 * sizeof(uint32_t), sizeof(uint32_t));
        assert(check == i);
        std::cout << "cluster-" << i << " has " << clu_size << " vectors,"
                  << " has clu_dim = " << clu_dim
                  << " clu_id_size = " << clu_id_size
                  << " clu_id_dim = " << clu_id_dim 
                  << std::endl;
        std::cout << "main meta: " << std::endl;
        std::cout << " number of vectors in per page: " << nvpp
                  << " number of pages 4 vector: " << npv
                  << " number of ids in per page: " << nipp
                  << " number of pages 4 id: " << npi
                  << std::endl;
        std::cout << "read old files' meta done." << std::endl;


        int npn = (clu_size - 1) / nnpp + 2;
        *(uint32_t*)(node_buf + 0 * sizeof(uint32_t)) = clu_size;
        *(uint32_t*)(node_buf + 1 * sizeof(uint32_t)) = clu_dim;
        *(uint32_t*)(node_buf + 2 * sizeof(uint32_t)) = page_size;
        *(uint32_t*)(node_buf + 3 * sizeof(uint32_t)) = nnpp;
        *(uint32_t*)(node_buf + 4 * sizeof(uint32_t)) = npn;
        *(uint32_t*)(node_buf + 5 * sizeof(uint32_t)) = i;
        data_writer[i].write(node_buf, page_size);



        int write_cnt = 0;
        for (int j = 1; j < npn; j ++) {
            memset(node_buf, 0, page_size);
            for (int k = 0; k < nnpp && write_cnt < clu_size; k ++) {
                if (0 == write_cnt % nvpp)
                    raw_data_file_handlers[i].read(dat_buf, page_size);
                if (0 == write_cnt % nipp)
                    ids_data_file_handlers[i].read(ids_buf, page_size);
                memcpy(node_buf + k * node_size, dat_buf + (write_cnt % nvpp) * vector_size, vector_size);
                memcpy(node_buf + k * node_size + vector_size, ids_buf + (write_cnt % nipp) * id_size, id_size);
                write_cnt ++;
            }
            data_writer[i].write(node_buf, page_size);
        }
    }


    for (auto i = 0; i < cluster_num; i ++) {
        raw_data_file_handlers[i].close();
        ids_data_file_handlers[i].close();        
        data_writer[i].close();
    }

    delete[] dat_buf;
    delete[] ids_buf;
    delete[] node_buf;
    return 0;
}
