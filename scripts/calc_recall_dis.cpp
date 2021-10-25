#include <stdio.h>

const char *flat = "bigann_flat.txt";
const char *index = "bigann_ivf_64.txt";

int nq = 1000;
int topk = 1000;
bool is_l2 = true;

int main() {
  FILE *f1 = fopen(flat, "r");
  FILE *f2 = fopen(index, "r");
  if (!f1 || !f2) {
    return 0;
  }

  int truth = 0;
  double dis_base, dis_cnt;
  for (int i = 0; i < nq; i++) {
    for (int j = 1; j < topk; j++)
      fscanf(f1, "%*lf%*d");
    fscanf(f1, "%lf%*d", &dis_base);

    for (int j = 0; j < topk; j++) {
      fscanf(f2, "%lf%*d", &dis_cnt);
      if (is_l2) {
        if (dis_cnt <= dis_base) {
          truth++;
        }
      } else {
        if (dis_cnt >= dis_base) {
          truth++;
        }
      }
    }
  }

  printf("recall rate %lf\n", double(truth) / (nq * topk));

  return 0;
}