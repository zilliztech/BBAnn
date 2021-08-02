// ----------------------------------------------------------------------------------------------------
#include <iostream>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
// ----------------------------------------------------------------------------------------------------
#include "util/io_perf.h"
// ----------------------------------------------------------------------------------------------------
void write_mmap_sample_data() {
    int fd;
    char ch;
    struct stat textfilestat;
    fd = open("MMAP_DATA.txt", O_CREAT|O_TRUNC|O_WRONLY, 0777);
    if (fd == -1) {
        perror("File open error ");
        return;
    }
    // Write A to Z
    ch = 'A';
    for (auto i = 0; i < 10'000'000; ++i) {
        write(fd, &ch, sizeof(ch));
    }
    close(fd);
    return;
}
// ----------------------------------------------------------------------------------------------------
int IO_function() {
    struct stat mmapstat;
    char *data;
    int minbyteindex;
    int maxbyteindex;
    int offset;
    int fd;
    int unmapstatus;
    write_mmap_sample_data();
    if (stat("MMAP_DATA.txt", &mmapstat) == -1) {
        perror("stat failure");
        return 1;
    }

    if ((fd = open("MMAP_DATA.txt", O_RDONLY)) == -1) {
        perror("open failure");
        return 1;
    }
    data = static_cast<char*>(mmap((caddr_t)0, mmapstat.st_size, PROT_READ, MAP_SHARED, fd, 0));

    if (data == (caddr_t)(-1)) {
        perror("mmap failure");
        return 1;
    }
    minbyteindex = 0;
    maxbyteindex = mmapstat.st_size - 1;

    unmapstatus = munmap(data, mmapstat.st_size);

    if (unmapstatus == -1) {
        perror("munmap failure");
        return 1;
    }
    close(fd);
    system("rm -f MMAP_DATA.txt");
}
// ----------------------------------------------------------------------------------------------------
int main() {
    {
#if IOPERF
        Syscr_Counter s;
#endif
        auto result = IO_function();
    }
    std::cout << "------------------------------------------------" << std::endl;
    {
#if IOPERF
        PID_IO_Counter s;
#endif
        auto result = IO_function();
    }
    std::cout << "------------------------------------------------" << std::endl;
    {
#if IOPERF
        DiskStat_Read_Counter s;
#endif
        auto result = IO_function();
    }
    return 0;
}
// ----------------------------------------------------------------------------------------------------

