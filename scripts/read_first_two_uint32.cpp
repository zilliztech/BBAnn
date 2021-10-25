#include <stdint.h>
#include <stdio.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Please enter file path as the first and the only command line "
           "argument\n");
    return 1;
  }

  uint32_t a, b;

  FILE *f = fopen(argv[1], "r");
  if (f == NULL) {
    fprintf(stderr, "Failed to open %s\n", argv[1]);
    return 1;
  }

  fread(&a, sizeof(uint32_t), 1, f);
  fread(&b, sizeof(uint32_t), 1, f);

  printf("Base data: n dim\n");
  printf("Query data: n dim\n");
  printf("Ground truth: nq topk\n");
  printf("====================\n");

  printf("%d %d\n", a, b);

  return 0;
}