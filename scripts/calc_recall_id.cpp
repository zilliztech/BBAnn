#include <algorithm>
#include <stdio.h>
#include <unordered_set>

const char *flat = "bigann_flat_nb_5M_topk_1K.txt";
// const char* index = "bigann_ivf_64.txt";
// const char* flat = "yandex_text_to_image_flat_nb_5M_topk_1K.txt";
// const char* index = "faiss_pq.txt";
const char *index = "bigann_pq.txt";

int nq = 1000;
int topk = 1000;
bool is_l2 = true;

int main() {
  FILE *f1 = fopen(flat, "r");
  FILE *f2 = fopen(index, "r");
  if (!f1 || !f2) {
    return 0;
  }

  std::unordered_set<int32_t> gt;
  double recalls = 0;
  int32_t id, cnt;

  for (int i = 0; i < nq; i++) {
    cnt = 0;
    for (int j = 1; j < topk; j++) {
      fscanf(f1, "%*lf%d", &id);
      gt.insert(id);
    }

    for (int j = 0; j < topk; j++) {
      fscanf(f2, "%*lf%d", &id);
      if (gt.find(id) != gt.end())
        ++cnt;
    }

    recalls += cnt * 1.0 / topk;
  }

  printf("recall rate %lf\n", recalls / nq);

  return 0;
}