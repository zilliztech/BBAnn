#include <random>
#include <iostream>
#include <sys/time.h>

void clean_page_cache() {
    sync();
    int fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
    if (fd == -1) std::cout << "Fail to open drop cache file." << std::endl;
    if (1 != write(fd, "3", 1)) std::cout << "Fail to write drop cache file." << std::endl;
    close(fd);
}

int L2sqr_PLAIN(uint8_t *a, uint8_t *b, size_t n) {
    size_t i = 0;
    size_t dis = 0, dif;
    switch(n & 7) {
        default:
            while (n > 7) {
                n -= 8; dif=(uint8_t)a[i]-(uint8_t)b[i]; dis+=dif*dif; i++;
                case 7: dif=(uint8_t)a[i]-(uint8_t)b[i]; dis+=dif*dif; i++;
                case 6: dif=(uint8_t)a[i]-(uint8_t)b[i]; dis+=dif*dif; i++;
                case 5: dif=(uint8_t)a[i]-(uint8_t)b[i]; dis+=dif*dif; i++;
                case 4: dif=(uint8_t)a[i]-(uint8_t)b[i]; dis+=dif*dif; i++;
                case 3: dif=(uint8_t)a[i]-(uint8_t)b[i]; dis+=dif*dif; i++;
                case 2: dif=(uint8_t)a[i]-(uint8_t)b[i]; dis+=dif*dif; i++;
                case 1: dif=(uint8_t)a[i]-(uint8_t)b[i]; dis+=dif*dif; i++;
            }
    }
    return dis;
}