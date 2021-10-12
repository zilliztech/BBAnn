#include "util/distance.h"
#include <sys/time.h>

using namespace std;

long n = 1000;
size_t dim = 128;

timeval t1, t2;
long int getTime(timeval end, timeval start) {
    return 1000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000;
}

int main() {
    float* xb = new float[n*dim];
    float* dis = new float[n*n];

    for (int i = 0; i < n*dim; ++i) {
        xb[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/10.0f));
    }
    printf("random done\n");

    gettimeofday(&t1, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          dis[i * n + j] = L2sqr<float, float, float>(xb + i * dim, xb + j * dim, dim);
        }
    }
    gettimeofday(&t2, 0);
    printf("cost %ld ms\n", getTime(t2, t1));

    delete[] xb;
    delete[] dis;

    return 0;
}