#include "util/utils.h"
#include <cstdlib>
#include <ctime>
#include <iostream>

int main() {
  int len = 100;
  srand((unsigned)time(NULL));
  std::vector<uint32_t> cids(len);
  std::vector<uint32_t> qids(len);
  std::vector<uint32_t> offs(len);
  for (auto i = 0; i < len; i++) {
    cids[i] = random() % 10 + 1;
    qids[i] = random() % 100 + 1;
    offs[i] = random() % 10000 + 1;
  }
  std::vector<uint64_t> rids(len);
  for (auto i = 0; i < len; i++) {
    rids[i] = gen_refine_id(cids[i], offs[i], qids[i]);
  }
  for (auto i = 0; i < len; i++) {
    uint32_t cid, off, qid;
    parse_refine_id(rids[i], cid, off, qid);
    assert(cid == cids[i]);
    assert(off == offs[i]);
    assert(qid == qids[i]);
  }
  return 0;
}
