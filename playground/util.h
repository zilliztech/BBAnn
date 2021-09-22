#include <random>
#include <iostream>
#include <sys/time.h>

template<typename T1, typename T2, typename R>
R IP(T1 *a, T2 *b, size_t n) {
    size_t i = 0;
    R dis = 0;
    switch(n & 7) {
        default:
            while (n > 7) {
                n -= 8; dis+=(R)a[i]*(R)b[i]; i++;
                case 7: dis+=(R)a[i]*(R)b[i]; i++;
                case 6: dis+=(R)a[i]*(R)b[i]; i++;
                case 5: dis+=(R)a[i]*(R)b[i]; i++;
                case 4: dis+=(R)a[i]*(R)b[i]; i++;
                case 3: dis+=(R)a[i]*(R)b[i]; i++;
                case 2: dis+=(R)a[i]*(R)b[i]; i++;
                case 1: dis+=(R)a[i]*(R)b[i]; i++;
            }
    }
    return dis;
}

int rand_int() {
    static std::mt19937 generator(3456);
    return generator() & 0x7fffffff;
}

long int getTime(timeval end, timeval start) {
    return 1000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000;
}

void clean_page_cache() {
    sync();
    int fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
    if (fd == -1) std::cout << "Fail to open drop cache file." << std::endl;
    if (1 != write(fd, "3", 1)) std::cout << "Fail to write drop cache file." << std::endl;
    close(fd);
}