// sudo apt-get install libaio-dev
// g++ test_different_reader.cpp -O3 -fopenmp -laio

#include <algorithm>
#include <errno.h>
#include <fcntl.h>
#include <libaio.h>
#include <omp.h>
#include <random>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
using namespace std;

const char *file_path = "../data/BIGANN/learn.100M.u8bin";

template <typename T1, typename T2, typename R> R IP(T1 *a, T2 *b, size_t n) {
  size_t i = 0;
  R dis = 0;
  switch (n & 7) {
  default:
    while (n > 7) {
      n -= 8;
      dis += (R)a[i] * (R)b[i];
      i++;
    case 7:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 6:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 5:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 4:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 3:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 2:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 1:
      dis += (R)a[i] * (R)b[i];
      i++;
    }
  }
  return dis;
}

int rand_int() {
  static std::mt19937 generator(3456);
  return generator() & 0x7fffffff;
}

timeval t1, t2;
long int getTime(timeval end, timeval start) {
  return 1000 * (end.tv_sec - start.tv_sec) +
         (end.tv_usec - start.tv_usec) / 1000;
}

int nq = 5000000;
int nb = 100000000;
int dim = 128;

int *xq = nullptr;
unsigned char *xb = nullptr;
int max_len = 0, min_len = 0x7fffffff;

int main() {
  xq = new int[nq];
  for (int i = 0; i < nq; i++) {
    xq[i] = rand_int() % nb;
  }
  // Sorting has a great influence
  sort(xq, xq + nq);

  gettimeofday(&t1, 0);

  /*
      // fread
      int thread_num = omp_get_max_threads();
      xb = new unsigned char[thread_num*dim];
  #pragma omp parallel reduction(max:max_len) reduction(min:min_len)
      {
          FILE *fi=fopen(file_path,"r");
          int rank = omp_get_thread_num();
          unsigned char *tmp = xb + rank * dim;
  #pragma omp for
          for (int i=0;i<nq;i++){
              fseek(fi, (long long)(xq[i]) * dim + 8,SEEK_SET);
              fread(tmp,1,dim,fi);
              int len = IP<unsigned char,unsigned char,int>(tmp,tmp,dim);
              if(max_len<len) {max_len=len;}
              if(min_len>len) {min_len=len;}
          }
          fclose(fi);
      }
  */

  /*
      // mmap
      {
          int fd = open(file_path, O_RDONLY | O_DIRECT);
          xb = (unsigned char*)mmap(NULL, (long long)nb * dim + 8, PROT_READ,
  MAP_SHARED, fd, 0); #pragma omp parallel for reduction(max:max_len)
  reduction(min:min_len) for (int i=0;i<nq;i++){ unsigned char *tmp = xb +
  ((long long)xq[i] * dim + 8); int len = IP<unsigned char,unsigned
  char,int>(tmp,tmp,dim); if(max_len<len) {max_len=len;} if(min_len>len)
  {min_len=len;}
          }
      }
  */

  // libaio
  const int MAX_EVENTS = 1024;

  int fd = open(file_path, O_RDONLY);
  xb = new unsigned char[MAX_EVENTS * dim];

  io_context_t ctx;
  memset(&ctx, 0, sizeof(ctx));
  io_setup(MAX_EVENTS, &ctx);

  struct iocb **p_cb = new iocb *[MAX_EVENTS];
  struct iocb *cb = new iocb[MAX_EVENTS];
  struct io_event *events = new io_event[MAX_EVENTS];

  for (int j = 0; j < MAX_EVENTS; j++) {
    p_cb[j] = cb + j;
  }

  for (int i = 0; i < nq; i += MAX_EVENTS) {
    int last_i = std::min(nq, i + MAX_EVENTS);
    int batch_num = last_i - i;
#pragma omp parallel for
    for (int j = 0; j < batch_num; j++) {
      io_prep_pread(p_cb[j], fd, xb + j * dim, dim,
                    (long long)(xq[i + j]) * dim + 8);
    }

    int rst = io_submit(ctx, batch_num, p_cb);
    // printf("io_submit %d\n", rst);
    rst = io_getevents(ctx, batch_num, batch_num, events, nullptr);
    // printf("io_getevents %d\n", rst);

#pragma omp parallel for reduction(max : max_len) reduction(min : min_len)
    for (int j = 0; j < batch_num; j++) {
      unsigned char *tmp = xb + j * dim;
      int len = IP<unsigned char, unsigned char, int>(tmp, tmp, dim);
      if (max_len < len) {
        max_len = len;
      }
      if (min_len > len) {
        min_len = len;
      }
    }
  }
  /**/

  gettimeofday(&t2, 0);
  printf("cost %ldms\n", getTime(t2, t1));

  printf("%d %d\n", max_len, min_len);

  return 0;
}
