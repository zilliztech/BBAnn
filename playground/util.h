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