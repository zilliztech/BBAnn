#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>

const char* output_file_path = "new_query";

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: ./generate_query_from_base [path] [n] [num_bytes_per_dim]\n");
        printf("Example:\n");
        printf("\t ./generate_query_from_base ../../data/BIGANN/base.1B.u8bin 1 1\n");
        return 1;
    }

    uint32_t n, dim;
    uint32_t new_n = atoi(argv[2]), bytes_per_dim = atoi(argv[3]);

    FILE* f = fopen(argv[1], "r");
    if (f == NULL) {
        fprintf(stderr, "Failed to open %s\n", argv[1]);
        return 1;
    }

    fread(&n, sizeof(uint32_t), 1, f);
    fread(&dim, sizeof(uint32_t), 1, f);

    FILE* fout = fopen(output_file_path, "w");
    if (fout == NULL) {
        fprintf(stderr, "Failed to write %s\n", output_file_path);
        return 1;
    }

    uint32_t buf_size = bytes_per_dim * dim;
    uint8_t* buf = new uint8_t[buf_size];

    fwrite(&new_n, sizeof(uint32_t), 1, fout);
    fwrite(&dim, sizeof(uint32_t), 1, fout);
    for (int i = 0; i < std::min(n, new_n); ++i) {
        fread(buf, sizeof(uint8_t), buf_size, f);
        fwrite(buf, sizeof(uint8_t), buf_size, fout);
    }

    fclose(f);
    fclose(fout);

    delete[] buf;

    return 0;
}