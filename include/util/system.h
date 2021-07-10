#pragma once

#include <cstdint>
#include <stdio.h>

int64_t get_L3_Size() {
    static int64_t l3_size = -1;
    constexpr int64_t KB = 1024;
    if (l3_size == -1) {
        l3_size = 12 * KB * KB; // 12M

        FILE* file = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size","r");
        int32_t result = 0;
        if (file){
            if (1 == fscanf(file, "%dK", &result)) {
                l3_size = result * KB;
            }
            fclose(file);
        }
    }
    return l3_size;
}