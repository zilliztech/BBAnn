#pragma once
#include <stddef.h>
#include <stdio.h>

// Data type: T1, T2
// Distance type: R

template<typename T1, typename T2, typename R>
R L2sqr(T1 *a, T2 *b, size_t n) {
    size_t i = 0;
    R dis = 0, dif;
    switch(n & 7) {
        default:
            while (n > 7) {
                n -= 8; dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 7: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 6: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 5: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 4: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 3: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 2: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 1: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
            }
    }
    return dis;
}

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





