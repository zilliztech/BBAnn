// ----------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
// ----------------------------------------------------------------------------------------------------
#include "util/io_perf.h"
#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <future>
#include <atomic>
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
void __attribute__((optimize("O1"))) write_mmap_sample_data() {
    int fd;
    fd = open("MMAP_DATA.txt", O_CREAT | O_TRUNC | O_WRONLY, 0777);
    if (fd == -1) {
        perror("File open error ");
        return;
    }
    char ch = 'A';
    int write_byte = 0;
    for (auto i = 0; i < 1'000; ++i) {
        write_byte = write(fd, &ch, sizeof(ch));
        assert(write_byte != -1);
    }
    close(fd);
    return;
}
//---------------------------------------------------------------------------
int IO_function() {
//    struct stat mmapstat;
//    char *data;
//    int minbyteindex;
//    int maxbyteindex;
//    int offset;
//    int fd;
//    int unmapstatus;
    write_mmap_sample_data();
//    if (stat("MMAP_DATA.txt", &mmapstat) == -1) {
//        perror("stat failure");
//        return 1;
//    }
//    if ((fd = open("MMAP_DATA.txt", O_RDONLY)) == -1) {
//        perror("open failure");
//        return 1;
//    }
//    data = static_cast<char*>(mmap((caddr_t)0, mmapstat.st_size, PROT_READ, MAP_SHARED, fd, 0));
//
//    if (data == (caddr_t)(-1)) {
//        perror("mmap failure!!");
//        return 1;
//    }
//    minbyteindex = 0;
//    maxbyteindex = mmapstat.st_size - 1;
//
//    unmapstatus = munmap(data, mmapstat.st_size);
//
//    if (unmapstatus == -1) {
//        perror("munmap failure");
//        return 1;
//    }
//    close(fd);
//    system("rm -f MMAP_DATA.txt");
    return 0;
}
//---------------------------------------------------------------------------
TEST(IOPerf, Basic) {
    // Let this thread only run 1s.
    std::thread fut([]() {
        DiskStat_Read_Counter s1;
        PID_IO_Counter s2;
        auto result = IO_function();
    });
    fut.detach();
    std::this_thread::sleep_for(std::chrono_literals::operator ""s(1));
    return;
}
//---------------------------------------------------------------------------
} // namespace
//---------------------------------------------------------------------------
