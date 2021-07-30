#pragma once
// ----------------------------------------------------------------------------------------------------
#include <unistd.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <array>
// ----------------------------------------------------------------------------------------------------
class Syscr_Counter {
 private:
    uint64_t pre_iamge = 0;

    /**
     * Need a super user permission to access: /proc/$PID/io
     * @return the number of read syscalls as an I/O counter.
     */
    uint64_t read_pid_syscr() {
        const pid_t pid = getpid();
        const auto pid_str = std::to_string(pid);
        const auto filename = "/proc/" + pid_str + "/io";
        std::ifstream file(filename);
        assert(!file.bad());
        std::string line;
        while (std::getline(file, line)) {
            if (line.substr(0, 7) == "syscr: ") {
                const auto number_str = line.substr(7);
                return std::stoull(number_str);
            }
        }
        __builtin_unreachable();
    }

 public:
    Syscr_Counter() : pre_iamge(read_pid_syscr()) {}
    ~Syscr_Counter() {
        std::cout << "# read syscall: " << read_pid_syscr() - pre_iamge << std::endl;
    }
};
// ----------------------------------------------------------------------------------------------------
/**
 * A class parsing a line of file /proc/diskstats
 * Need a super user permission to access: /proc/diskstats
 */
class DiskStat {
 private:
    std::string device_name;
    std::array<uint64_t, 18> fields;  // fields[0] is left unused.
    // Field  1 -- # of reads completed
    // Field  2 -- # of reads merged
    // Field  3 -- # of sectors read
    // Field  4 -- # of milliseconds spent reading

    // Field  5 -- # of writes completed
    // Field  6 -- # of writes merged
    // Field  7 -- # of sectors written
    // Field  8 -- # of milliseconds spent writing

    // Field  9 -- # of I/Os currently in progress
    // Field 10 -- # of milliseconds spent doing I/Os
    // Field 11 -- weighted # of milliseconds spent doing I/Os

    // Field 12 -- # of discards completed
    // Field 13 -- # of discards merged
    // Field 14 -- # of sectors discarded
    // Field 15 -- # of milliseconds spent discarding

    // Field 16 -- # of flush requests completed
    // Field 17 -- # of milliseconds spent flushing

    /**
     * Private helper function. The only way to construct a DiskStat.
     * @param info a line of file /proc/diskstats
     */
    DiskStat(std::string device_name, std::string info) {
        const auto first_space = info.find(' ');
        auto main_str = info.substr(first_space + 1);
        assert(main_str.front() != ' ');
        this->device_name = info.substr(0, first_space);
        assert(device_name == this->device_name);
        // You should have a Kernel  5.5+ to reach all 17 fields.
        for (auto i = 1; i < 18; ++i) {
            const auto next_space = main_str.find(' ');
            fields[i] = std::stoull(main_str.substr(0, next_space));
            main_str = main_str.substr(next_space + 1);
        }
    }

    DiskStat() = delete;

 public:
    /**
     * Read /proc/diskstats to get disk's info.
     * Need a super user permission to access: /proc/diskstats
     * @return
     */
    static DiskStat read_IO_DiskStat(std::string device_name = "nvme0n1") {
        const auto filename = "/proc/diskstats";
        std::ifstream file(filename);
        assert(!file.bad());
        std::string line;
        while (std::getline(file, line)) {
            const auto n = line.find(device_name);
            if (n != std::string::npos) {
                const auto io_str = line.substr(n);
                return DiskStat(device_name, io_str);
            }
        }
        __builtin_unreachable(); // Wrong device name.
    }

    /**
     * @param field_index
     * @return get a field
     */
    uint64_t get_nth_field(int field_index) const {
        assert(field_index != 0);
        return fields.at(field_index);
    }

    /**
     * @param field_index
     * @return get a field
     */
    uint64_t operator[](int field_index) const {
        assert(field_index != 0);
        return fields.at(field_index);
    }
};
// ----------------------------------------------------------------------------------------------------
class DiskStat_Read_Counter {
 private:
    DiskStat pre_iamge;

 public:
    DiskStat_Read_Counter() : pre_iamge(DiskStat::read_IO_DiskStat()) {}

    ~DiskStat_Read_Counter() {
        DiskStat post_iamge = DiskStat::read_IO_DiskStat();
        std::cout << "# read completed: " << post_iamge[1] - pre_iamge[1] << std::endl;
        std::cout << "# read merged: " << post_iamge[2] - pre_iamge[2] << std::endl;
        std::cout << "# sectors read: " << post_iamge[3] - pre_iamge[3] << std::endl;
        std::cout << "# milliseconds spent reading: " << post_iamge[4] - pre_iamge[4] << std::endl;
        std::cout << "# of milliseconds spent doing I/Os: " << post_iamge[10] - pre_iamge[10] << std::endl;
        std::cout << "weighted # of milliseconds spent doing I/Os: " << post_iamge[11] - pre_iamge[11] << std::endl;

    }
};
// ----------------------------------------------------------------------------------------------------
