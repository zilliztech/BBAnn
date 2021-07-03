#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

const char* output_file_path = "new_base_dataset";
const int BATCH_SIZE = 1e6;

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: ./get_first_n_from_base [path] [new_n_in_M] [num_bytes_per_dim]\n");
        printf("Example:\n");
        printf("\t ./get_first_n_from_base ../../data/BIGANN/base.1B.u8bin 10 1\n");
        return 1;
    }

    uint32_t n, dim;
    uint32_t new_n = atoi(argv[2])*1e6, bytes_per_dim = atoi(argv[3]);

    if (new_n % BATCH_SIZE != 0) {
        fprintf(stderr, "New n %d is not divisible by %d\n", new_n, BATCH_SIZE);
        return 1;
    }

    FILE* f = fopen(argv[1], "r");
    if (f == NULL) {
        fprintf(stderr, "Failed to open %s\n", argv[1]);
        return 1;
    }

    fread(&n, sizeof(uint32_t), 1, f);
    fread(&dim, sizeof(uint32_t), 1, f);

    if (n < new_n) {
        printf("The new n %d is larger than the original n %d\n", new_n, n);
        return 1;
    }
    
    FILE* fout = fopen(output_file_path, "w");
    if (fout == NULL) {
        fprintf(stderr, "Failed to write %s\n", output_file_path);
        return 1;
    }

    uint32_t buf_size = BATCH_SIZE * bytes_per_dim * dim;
    uint8_t* buf = new uint8_t[buf_size];

    fwrite(&new_n, sizeof(uint32_t), 1, fout);
    fwrite(&dim, sizeof(uint32_t), 1, fout);
    for (int i = 0; i < new_n/BATCH_SIZE; ++i) {
        fread(buf, sizeof(uint8_t), buf_size, f);
        fwrite(buf, sizeof(uint8_t), buf_size, fout);
    }

    fclose(f);
    fclose(fout);

    delete[] buf;

    return 0;
}