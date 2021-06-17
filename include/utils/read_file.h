#pragma once

#include <cstdint>
#include <stdio.h>

FILE* read_file_head (const char *fname, int32_t *n_out, int32_t *d_out) {
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        return nullptr;
    }

    fread(n_out, sizeof(int), 1, f);
    fread(d_out, sizeof(int), 1, f);

    return f;
}

template<typename T>
int32_t read_file_data (FILE *f, int32_t batch, int32_t dim, T* buff) {
    int32_t batch_read = fread(buff, sizeof(T) * dim, batch, f);
    return batch_read;
}