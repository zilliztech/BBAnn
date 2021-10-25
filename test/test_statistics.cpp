#include "util/statistics.h"
#include <stdio.h>
#include <string.h>

enum Type { FLOAT, UINT8, INT8, UNKNOWN };
int Type_Size[] = {4, 1, 1, 0};

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("error argc %d", argc);
    return -1;
  }

  Type t = UNKNOWN;
  int len = strlen(argv[1]);
  if (len > 5) {
    if (strcmp(argv[1] + len - 5, ".fbin") == 0) {
      t = FLOAT;
    } else if (strcmp(argv[1] + len - 5, "u8bin") == 0) {
      t = UINT8;
    } else if (strcmp(argv[1] + len - 5, "i8bin") == 0) {
      t = INT8;
    } else {
    }
  }
  if (t == UNKNOWN) {
    printf("error argv[1] %s", argv[1]);
    return -2;
  }

  FILE *fi = fopen(argv[1], "r");
  if (fi == nullptr) {
    printf("could not open %s\n", argv[1]);
    return -3;
  }

  int meta[2]; // meta[0]: n; meta[1]: dim
  fread(meta, sizeof(int), 2, fi);

  int n = meta[0];
  int dim = meta[1];

  const int BATCH = 1000000;
  void *buff = malloc(BATCH * dim * Type_Size[t]);
  if (buff == nullptr)
    printf("mall error\n");

  double global_avg_len = 0;
  double global_max_len = 0;
  double global_min_len = std::numeric_limits<double>::max();

  for (int i = 0, batch_cnt = 0; i < n; i += BATCH, batch_cnt++) {
    int loacl_batch = (i + BATCH < n) ? BATCH : (n - i);
    // ignore the remain data
    if (loacl_batch != BATCH && i != 0)
      break;

    fread(buff, loacl_batch * dim, Type_Size[t], fi);

    double avg_len, max_len, min_len;
    if (t == FLOAT) {
      stat_length<float>((float *)buff, loacl_batch, dim, max_len, min_len,
                         avg_len);
    } else if (t == UINT8) {
      stat_length<unsigned char>((unsigned char *)buff, loacl_batch, dim,
                                 max_len, min_len, avg_len);
    } else if (t == INT8) {
      stat_length<char>((char *)buff, loacl_batch, dim, max_len, min_len,
                        avg_len);
    }

    printf("batch_i %d avg %lf min %lf max %lf\n", batch_cnt, avg_len, min_len,
           max_len);

    if (max_len > global_max_len)
      global_max_len = max_len;
    if (min_len < global_min_len)
      global_min_len = min_len;
    global_avg_len = (global_avg_len * batch_cnt + avg_len) / (batch_cnt + 1);
  }

  printf("stat %d * %d vectors:\n", n / BATCH * BATCH, dim);
  printf("avg len %lf\nmin len %lf\nmax len %lf\n", global_avg_len,
         global_min_len, global_max_len);

  fclose(fi);
  return 0;
}